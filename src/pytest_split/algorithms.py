import enum
import heapq
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from _pytest import nodes


class TestGroup(NamedTuple):
    selected: "list[nodes.Item]"
    deselected: "list[nodes.Item]"
    duration: float


class AlgorithmBase(ABC):
    """Abstract base class for the algorithm implementations."""

    @abstractmethod
    def __call__(
        self, splits: int, items: "list[nodes.Item]", durations: "dict[str, float]"
    ) -> "list[TestGroup]":
        pass

    def __hash__(self) -> int:
        return hash(self.__class__.__name__)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlgorithmBase):
            return NotImplemented
        return self.__class__.__name__ == other.__class__.__name__


class LeastDurationAlgorithm(AlgorithmBase):
    """
    Split tests into groups by runtime.
    It walks the test items, starting with the test with largest duration.
    It assigns the test with the largest runtime to the group with the smallest duration sum.

    The algorithm sorts the items by their duration. Since the sorting algorithm is stable, ties will be broken by
    maintaining the original order of items. It is therefore important that the order of items be identical on all nodes
    that use this plugin. Due to issue #25 this might not always be the case.

    :param splits: How many groups we're splitting in.
    :param items: Test items passed down by Pytest.
    :param durations: Our cached test runtimes. Assumes contains timings only of relevant tests
    :return:
        List of groups
    """

    def __call__(
        self, splits: int, items: "list[nodes.Item]", durations: "dict[str, float]"
    ) -> "list[TestGroup]":
        items_with_durations = _get_items_with_durations(items, durations)

        # add index of item in list
        items_with_durations_indexed = [
            (*tup, i) for i, tup in enumerate(items_with_durations)
        ]

        # Sort by name to ensure it's always the same order
        items_with_durations_indexed = sorted(
            items_with_durations_indexed, key=lambda tup: str(tup[0])
        )

        # sort in ascending order
        sorted_items_with_durations = sorted(
            items_with_durations_indexed, key=lambda tup: tup[1], reverse=True
        )

        selected: list[list[tuple[nodes.Item, int]]] = [[] for _ in range(splits)]
        deselected: list[list[nodes.Item]] = [[] for _ in range(splits)]
        duration: list[float] = [0 for _ in range(splits)]

        # create a heap of the form (summed_durations, group_index)
        heap: list[tuple[float, int]] = [(0, i) for i in range(splits)]
        heapq.heapify(heap)
        for item, item_duration, original_index in sorted_items_with_durations:
            # get group with smallest sum
            summed_durations, group_idx = heapq.heappop(heap)
            new_group_durations = summed_durations + item_duration

            # store assignment
            selected[group_idx].append((item, original_index))
            duration[group_idx] = new_group_durations
            for i in range(splits):
                if i != group_idx:
                    deselected[i].append(item)

            # store new duration - in case of ties it sorts by the group_idx
            heapq.heappush(heap, (new_group_durations, group_idx))

        groups = []
        for i in range(splits):
            # sort the items by their original index to maintain relative ordering
            # we don't care about the order of deselected items
            s = [
                item
                for item, original_index in sorted(selected[i], key=lambda tup: tup[1])
            ]
            group = TestGroup(
                selected=s, deselected=deselected[i], duration=duration[i]
            )
            groups.append(group)
        return groups


class DurationBasedChunksAlgorithm(AlgorithmBase):
    """
    Split tests into groups by runtime.
    Ensures tests are split into non-overlapping groups.
    The original list of test items is split into groups by finding boundary indices i_0, i_1, i_2
    and creating group_1 = items[0:i_0], group_2 = items[i_0, i_1], group_3 = items[i_1, i_2], ...

    :param splits: How many groups we're splitting in.
    :param items: Test items passed down by Pytest.
    :param durations: Our cached test runtimes. Assumes contains timings only of relevant tests
    :return: List of TestGroup
    """

    def __call__(
        self, splits: int, items: "list[nodes.Item]", durations: "dict[str, float]"
    ) -> "list[TestGroup]":
        items_with_durations = _get_items_with_durations(items, durations)
        time_per_group = sum(map(itemgetter(1), items_with_durations)) / splits

        selected: list[list[nodes.Item]] = [[] for i in range(splits)]
        deselected: list[list[nodes.Item]] = [[] for i in range(splits)]
        duration: list[float] = [0 for i in range(splits)]

        group_idx = 0
        for item, item_duration in items_with_durations:
            if duration[group_idx] >= time_per_group:
                group_idx += 1

            selected[group_idx].append(item)
            for i in range(splits):
                if i != group_idx:
                    deselected[i].append(item)
            duration[group_idx] += item_duration

        return [
            TestGroup(
                selected=selected[i], deselected=deselected[i], duration=duration[i]
            )
            for i in range(splits)
        ]


def _get_items_with_durations(
    items: "list[nodes.Item]", durations: "dict[str, float]"
) -> "list[tuple[nodes.Item, float]]":
    durations = _remove_irrelevant_durations(items, durations)
    avg_duration_per_test = _get_avg_duration_per_test(durations)
    items_with_durations = [
        (item, durations.get(item.nodeid, avg_duration_per_test)) for item in items
    ]
    return items_with_durations


def _get_avg_duration_per_test(durations: "dict[str, float]") -> float:
    if durations:
        avg_duration_per_test = sum(durations.values()) / len(durations)
    else:
        # If there are no durations, give every test the same arbitrary value
        avg_duration_per_test = 1
    return avg_duration_per_test


def _remove_irrelevant_durations(
    items: "list[nodes.Item]", durations: "dict[str, float]"
) -> "dict[str, float]":
    # Filtering down durations to relevant ones ensures the avg isn't skewed by irrelevant data
    test_ids = [item.nodeid for item in items]
    durations = {name: durations[name] for name in test_ids if name in durations}
    return durations


class ScopeAwareLeastDurationAlgorithm(AlgorithmBase):
    """
    Split tests into groups by runtime, keeping tests from the same scope
    (module/class) together.

    This algorithm first aggregates tests by their scope (module path for
    ``--dist=loadscope``), computes the total duration per scope, then uses
    a greedy bin-packing algorithm (like LeastDurationAlgorithm) to assign
    *scopes* (not individual tests) to groups.

    This combines the duration-balancing property of LeastDurationAlgorithm
    with the locality-preserving property of DurationBasedChunksAlgorithm.
    When used with ``pytest-xdist --dist=loadscope``, it avoids redundant
    module imports and fixture setups across workers.

    If a single scope exceeds the ideal time-per-group, it is further
    subdivided by class (``module::Class``) so that sub-scopes can be
    distributed across multiple groups.  This handles pathologically
    large test modules while still preserving locality at the class level.

    :param splits: How many groups we're splitting in.
    :param items: Test items passed down by Pytest.
    :param durations: Our cached test runtimes.
    :return: List of groups
    """

    def __call__(
        self, splits: int, items: "list[nodes.Item]", durations: "dict[str, float]"
    ) -> "list[TestGroup]":
        items_with_durations = _get_items_with_durations(items, durations)
        total_duration = sum(dur for _, dur in items_with_durations)
        ideal_per_group = total_duration / splits if splits else total_duration

        # Group items by module scope (everything before the first ::)
        module_items: dict[str, list[tuple[nodes.Item, float, int]]] = {}
        for orig_idx, (item, dur) in enumerate(items_with_durations):
            module = item.nodeid.split("::")[0]
            module_items.setdefault(module, []).append((item, dur, orig_idx))

        # Compute total duration per module
        module_durations = {
            module: sum(dur for _, dur, _ in items_list)
            for module, items_list in module_items.items()
        }

        # Build scope units: for modules that exceed ideal_per_group,
        # subdivide by class to allow finer-grained distribution.
        # If a class-level scope still exceeds ideal, fall back to
        # individual test packing for that scope.
        # scope_key -> list of (item, dur, orig_idx)
        scope_items: dict[str, list[tuple[nodes.Item, float, int]]] = {}

        for module, mod_dur in module_durations.items():
            if mod_dur > ideal_per_group:
                # Subdivide by class: use "module::Class" or "module::<no-class>"
                class_items: dict[str, list[tuple[nodes.Item, float, int]]] = {}
                for item, dur, orig_idx in module_items[module]:
                    parts = item.nodeid.split("::")
                    if len(parts) >= 3:
                        # module::Class::method... -> scope is module::Class
                        class_key = f"{parts[0]}::{parts[1]}"
                    else:
                        # module::function -> scope is module::<no-class>
                        class_key = f"{parts[0]}::<no-class>"
                    class_items.setdefault(class_key, []).append((item, dur, orig_idx))
                # Check if any class scope still exceeds ideal
                for cls_key, cls_tests in class_items.items():
                    cls_dur = sum(dur for _, dur, _ in cls_tests)
                    if cls_dur > ideal_per_group:
                        # Fall back to individual test packing
                        for item, dur, orig_idx in cls_tests:
                            scope_items[item.nodeid] = [(item, dur, orig_idx)]
                    else:
                        scope_items[cls_key] = cls_tests
            else:
                scope_items[module] = module_items[module]

        # Compute total duration per scope unit
        scope_durations = {
            scope: sum(dur for _, dur, _ in items_list)
            for scope, items_list in scope_items.items()
        }

        # Sort scopes by name for determinism, then by duration descending
        sorted_scopes = sorted(scope_durations.keys())
        sorted_scopes = sorted(sorted_scopes, key=lambda s: scope_durations[s], reverse=True)

        # Greedy bin-packing of scopes into groups
        scope_to_group: dict[str, int] = {}
        heap: list[tuple[float, int]] = [(0.0, i) for i in range(splits)]
        heapq.heapify(heap)

        for scope in sorted_scopes:
            total_dur, group_idx = heapq.heappop(heap)
            scope_to_group[scope] = group_idx
            heapq.heappush(heap, (total_dur + scope_durations[scope], group_idx))

        # Assign items to groups, preserving original order within each group
        selected: list[list[tuple[nodes.Item, int]]] = [[] for _ in range(splits)]
        deselected: list[list[nodes.Item]] = [[] for _ in range(splits)]
        duration: list[float] = [0.0] * splits

        for scope, items_list in scope_items.items():
            group_idx = scope_to_group[scope]
            for item, dur, orig_idx in items_list:
                selected[group_idx].append((item, orig_idx))
                duration[group_idx] += dur
                for i in range(splits):
                    if i != group_idx:
                        deselected[i].append(item)

        groups = []
        for i in range(splits):
            s = [
                item
                for item, original_index in sorted(selected[i], key=lambda tup: tup[1])
            ]
            group = TestGroup(
                selected=s, deselected=deselected[i], duration=duration[i]
            )
            groups.append(group)
        return groups


class Algorithms(enum.Enum):
    duration_based_chunks = DurationBasedChunksAlgorithm()
    least_duration = LeastDurationAlgorithm()
    least_duration_by_scope = ScopeAwareLeastDurationAlgorithm()

    @staticmethod
    def names() -> "list[str]":
        return [x.name for x in Algorithms]
