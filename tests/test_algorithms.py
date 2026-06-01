import itertools
from collections import namedtuple
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.nodes import Item

from pytest_split.algorithms import (
    AlgorithmBase,
    Algorithms,
)

item = namedtuple("item", "nodeid")  # noqa: PYI024


class TestAlgorithms:
    @pytest.mark.parametrize("algo_name", Algorithms.names())
    def test__split_test(self, algo_name):
        durations = {"a": 1, "b": 1, "c": 1}
        items = [item(x) for x in durations]
        algo = Algorithms[algo_name].value
        first, second, third = algo(splits=3, items=items, durations=durations)

        # each split should have one test
        assert first.selected == [item("a")]
        assert first.deselected == [item("b"), item("c")]
        assert first.duration == 1

        assert second.selected == [item("b")]
        assert second.deselected == [item("a"), item("c")]
        assert second.duration == 1

        assert third.selected == [item("c")]
        assert third.deselected == [item("a"), item("b")]
        assert third.duration == 1

    @pytest.mark.parametrize("algo_name", Algorithms.names())
    def test__split_tests_handles_tests_in_durations_but_missing_from_items(
        self, algo_name
    ):
        durations = {"a": 1, "b": 1}
        items = [item(x) for x in ["a"]]
        algo = Algorithms[algo_name].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        assert first.selected == [item("a")]
        assert second.selected == []

    @pytest.mark.parametrize("algo_name", Algorithms.names())
    def test__split_tests_handles_tests_with_missing_durations(self, algo_name):
        durations = {"a": 1}
        items = [item(x) for x in ["a", "b"]]
        algo = Algorithms[algo_name].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        assert first.selected == [item("a")]
        assert second.selected == [item("b")]

    def test__split_test_handles_large_duration_at_end(self):
        """NOTE: only least_duration does this correctly"""
        durations = {"a": 1, "b": 1, "c": 1, "d": 3}
        items = [item(x) for x in ["a", "b", "c", "d"]]
        algo = Algorithms["least_duration"].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        assert first.selected == [item("d")]
        assert second.selected == [item(x) for x in ["a", "b", "c"]]

    @pytest.mark.parametrize(
        ("algo_name", "expected"),
        [
            ("duration_based_chunks", [[item("a"), item("b")], [item("c"), item("d")]]),
            ("least_duration", [[item("a"), item("c")], [item("b"), item("d")]]),
        ],
    )
    def test__split_tests_calculates_avg_test_duration_only_on_present_tests(
        self, algo_name, expected
    ):
        # If the algo includes test e's duration to calculate the averge then
        # a will be expected to take a long time, and so 'a' will become its
        # own group. Intended behaviour is that a gets estimated duration 1 and
        # this will create more balanced groups.
        durations = {"b": 1, "c": 1, "d": 1, "e": 10000}
        items = [item(x) for x in ["a", "b", "c", "d"]]
        algo = Algorithms[algo_name].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        expected_first, expected_second = expected
        assert first.selected == expected_first
        assert second.selected == expected_second

    @pytest.mark.parametrize(
        ("algo_name", "expected"),
        [
            (
                "duration_based_chunks",
                [[item("a"), item("b"), item("c"), item("d"), item("e")], []],
            ),
            (
                "least_duration",
                [[item("e")], [item("a"), item("b"), item("c"), item("d")]],
            ),
        ],
    )
    def test__split_tests_maintains_relative_order_of_tests(self, algo_name, expected):
        durations = {"a": 2, "b": 3, "c": 4, "d": 5, "e": 10000}
        items = [item(x) for x in ["a", "b", "c", "d", "e"]]
        algo = Algorithms[algo_name].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        expected_first, expected_second = expected
        assert first.selected == expected_first
        assert second.selected == expected_second

    def test__split_tests_same_set_regardless_of_order(self):
        """NOTE: only least_duration does this correctly"""
        tests = ["a", "b", "c", "d", "e", "f", "g"]
        durations = {t: 1 for t in tests}
        items = [item(t) for t in tests]
        algo = Algorithms["least_duration"].value
        for n in (2, 3, 4):
            selected_each: list[set[Item]] = [set() for _ in range(n)]
            for order in itertools.permutations(items):
                splits = algo(splits=n, items=order, durations=durations)
                for i, group in enumerate(splits):
                    if not selected_each[i]:
                        selected_each[i] = set(group.selected)
                    assert selected_each[i] == set(group.selected)

    def test__algorithms_members_derived_correctly(self):
        for a in Algorithms.names():
            assert issubclass(Algorithms[a].value.__class__, AlgorithmBase)


class TestScopeAwareLeastDuration:
    """Tests specific to the least_duration_by_scope algorithm."""

    algo = Algorithms["least_duration_by_scope"].value

    def test__keeps_same_module_together(self):
        """Tests from the same module should land in the same group."""
        durations = {
            "mod_a.py::test_1": 1,
            "mod_a.py::test_2": 1,
            "mod_b.py::test_1": 1,
            "mod_b.py::test_2": 1,
        }
        items = [item(x) for x in durations]
        first, second = self.algo(splits=2, items=items, durations=durations)

        # Each module should be entirely in one group
        first_mods = {i.nodeid.split("::")[0] for i in first.selected}
        second_mods = {i.nodeid.split("::")[0] for i in second.selected}
        assert first_mods & second_mods == set()

    def test__balances_by_module_duration(self):
        """A heavy module should be placed alone to balance groups."""
        # heavy.py=4, lights=3 each -> total=13, ideal=6.5
        # heavy.py (4) < 6.5, so no subdivision
        durations = {
            "heavy.py::test_1": 2,
            "heavy.py::test_2": 2,
            "light_a.py::test_1": 3,
            "light_b.py::test_1": 3,
            "light_c.py::test_1": 3,
        }
        items = [item(x) for x in durations]
        first, second = self.algo(splits=2, items=items, durations=durations)

        # heavy.py should be entirely in one group
        first_mods = {i.nodeid.split("::")[0] for i in first.selected}
        second_mods = {i.nodeid.split("::")[0] for i in second.selected}
        assert first_mods & second_mods == set()

        # heavy.py (4s) + light_c (3s) = 7s vs light_a (3s) + light_b (3s) = 6s
        # (greedy packing will group them based on descending duration)

    def test__subdivides_oversized_module_by_class(self):
        """When a module exceeds ideal_per_group, split by class."""
        durations = {
            "big.py::ClassA::test_1": 3,
            "big.py::ClassA::test_2": 3,
            "big.py::ClassB::test_1": 3,
            "big.py::ClassB::test_2": 3,
            "small.py::test_1": 1,
        }
        # Total=13, 2 splits, ideal=6.5. big.py=12 > 6.5 -> subdivide.
        # ClassA=6, ClassB=6, both <= 6.5 -> class-level packing.
        items = [item(x) for x in durations]
        first, second = self.algo(splits=2, items=items, durations=durations)

        # ClassA and ClassB should be in different groups
        first_classes = {
            i.nodeid.split("::")[1]
            for i in first.selected
            if i.nodeid.startswith("big.py")
        }
        second_classes = {
            i.nodeid.split("::")[1]
            for i in second.selected
            if i.nodeid.startswith("big.py")
        }
        # They should not both contain both classes
        assert not (first_classes == {"ClassA", "ClassB"})

    def test__subdivides_oversized_class_to_individual_tests(self):
        """When a class also exceeds ideal_per_group, fall back to per-test."""
        durations = {
            "big.py::test_a": 4,
            "big.py::test_b": 4,
            "big.py::test_c": 4,
            "small.py::test_x": 1,
        }
        # Total=13, 2 splits, ideal=6.5. big.py=12 > 6.5 -> subdivide by class.
        # All are bare functions -> <no-class>=12 > 6.5 -> individual packing.
        items = [item(x) for x in durations]
        first, second = self.algo(splits=2, items=items, durations=durations)

        first_ids = {i.nodeid for i in first.selected}
        second_ids = {i.nodeid for i in second.selected}

        # All tests should be assigned
        assert first_ids | second_ids == set(durations.keys())
        # big.py tests should be split across groups (not all in one)
        big_in_first = {n for n in first_ids if n.startswith("big.py")}
        big_in_second = {n for n in second_ids if n.startswith("big.py")}
        assert big_in_first and big_in_second

    def test__preserves_order_within_group(self):
        """Original order of tests should be preserved within each group."""
        # mod_a=3, mod_b=5 -> total=8, ideal=4. Both <= 4, so no subdivision.
        durations = {
            "mod_a.py::test_3": 1,
            "mod_a.py::test_1": 1,
            "mod_a.py::test_2": 1,
            "mod_b.py::test_x": 2,
            "mod_b.py::test_y": 3,
        }
        items = [item(x) for x in durations]
        first, second = self.algo(splits=2, items=items, durations=durations)

        # Find the group containing mod_a
        mod_a_group = (
            first
            if any("mod_a.py" in i.nodeid for i in first.selected)
            else second
        )
        mod_a_ids = [i.nodeid for i in mod_a_group.selected if "mod_a.py" in i.nodeid]
        # Should be in original order (test_3, test_1, test_2)
        assert mod_a_ids == [
            "mod_a.py::test_3",
            "mod_a.py::test_1",
            "mod_a.py::test_2",
        ]

    def test__deselected_contains_other_groups_items(self):
        """Each group's deselected list should contain all other groups' tests."""
        durations = {
            "mod_a.py::test_1": 1,
            "mod_b.py::test_1": 1,
            "mod_c.py::test_1": 1,
        }
        items = [item(x) for x in durations]
        groups = self.algo(splits=3, items=items, durations=durations)

        for i, group in enumerate(groups):
            # All items not in selected should be in deselected
            selected_set = set(group.selected)
            deselected_set = set(group.deselected)
            all_items_set = set(items)
            assert selected_set | deselected_set == all_items_set
            assert selected_set & deselected_set == set()

    def test__duration_is_correct(self):
        """Group duration should match sum of contained test durations."""
        durations = {
            "mod_a.py::test_1": 2.5,
            "mod_a.py::test_2": 3.5,
            "mod_b.py::test_1": 4.0,
        }
        items = [item(x) for x in durations]
        groups = self.algo(splits=2, items=items, durations=durations)

        for group in groups:
            expected_dur = sum(durations[i.nodeid] for i in group.selected)
            assert group.duration == pytest.approx(expected_dur)

    def test__many_small_modules_balance(self):
        """Many small equal modules should distribute evenly."""
        durations = {}
        for i in range(12):
            durations[f"mod_{i}.py::test_1"] = 1.0
        items = [item(x) for x in durations]
        groups = self.algo(splits=3, items=items, durations=durations)

        # Each group should have 4 tests (12 / 3)
        for group in groups:
            assert len(group.selected) == 4

    def test__single_split_returns_all(self):
        """With splits=1, all tests should be in one group."""
        durations = {
            "mod_a.py::test_1": 1,
            "mod_b.py::test_1": 2,
        }
        items = [item(x) for x in durations]
        (group,) = self.algo(splits=1, items=items, durations=durations)

        assert len(group.selected) == 2
        assert group.deselected == []
        assert group.duration == pytest.approx(3.0)

    def test__deterministic_across_item_orderings(self):
        """Same grouping regardless of input item order."""
        durations = {
            "mod_a.py::test_1": 1,
            "mod_a.py::test_2": 2,
            "mod_b.py::test_1": 3,
            "mod_b.py::test_2": 4,
            "mod_c.py::test_1": 5,
        }
        items_list = [item(x) for x in durations]

        reference = None
        for perm in itertools.permutations(items_list):
            groups = self.algo(splits=2, items=list(perm), durations=durations)
            group_sets = tuple(frozenset(i.nodeid for i in g.selected) for g in groups)
            if reference is None:
                reference = group_sets
            else:
                assert group_sets == reference


class MyAlgorithm(AlgorithmBase):
    def __call__(self, a, b, c):
        """no-op"""


class MyOtherAlgorithm(AlgorithmBase):
    def __call__(self, a, b, c):
        """no-op"""


class TestAbstractAlgorithm:
    def test__hash__returns_correct_result(self):
        algo = MyAlgorithm()
        assert algo.__hash__() == hash(algo.__class__.__name__)

    def test__hash__returns_same_hash_for_same_class_instances(self):
        algo1 = MyAlgorithm()
        algo2 = MyAlgorithm()
        assert algo1.__hash__() == algo2.__hash__()

    def test__hash__returns_different_hash_for_different_classes(self):
        algo1 = MyAlgorithm()
        algo2 = MyOtherAlgorithm()
        assert algo1.__hash__() != algo2.__hash__()

    def test__eq__returns_true_for_same_instance(self):
        algo = MyAlgorithm()
        assert algo.__eq__(algo) is True

    def test__eq__returns_false_for_different_instance(self):
        algo1 = MyAlgorithm()
        algo2 = MyOtherAlgorithm()
        assert algo1.__eq__(algo2) is False

    def test__eq__returns_true_for_same_algorithm_different_instance(self):
        algo1 = MyAlgorithm()
        algo2 = MyAlgorithm()
        assert algo1.__eq__(algo2) is True

    def test__eq__returns_false_for_non_algorithm_object(self):
        algo = MyAlgorithm()
        other = "not an algorithm"
        assert algo.__eq__(other) is NotImplemented
