# Location: tests/test_model_collection.py
"""
Tests for the SolutionSet and SolutionSetCollection classes.

These tests verify that Pareto fronts can be converted to SolutionSets,
and that SolutionSetCollections correctly store, retrieve, and manage
multiple sets.
"""

import pytest
from src.domain.core.paretofront import ParetoFront
from src.domain.core.solution import Solution
from src.domain.core.collection import SolutionSet, SolutionSetCollection


def test_pareto_front_to_solution_set():
    """
    Test that a ParetoFront can be converted to a SolutionSet with metadata.

    The conversion should preserve the name, algorithm, and the number of solutions.
    """
    sol1 = Solution(id="s1", selected_ids={"1"}, objectives={"satisfaction": 10, "effort": 2})
    front = ParetoFront([sol1])

    s_set = front.to_solution_set(name="Run 1", algorithm="NSGA2")

    assert isinstance(s_set, SolutionSet)
    assert s_set.name == "Run 1"
    assert s_set.algorithm == "NSGA2"
    assert len(s_set) == 1


def test_solution_set_collection():
    """
    Test that SolutionSetCollection correctly stores and retrieves multiple sets.

    The collection should:
        - Have the correct length (number of sets).
        - Allow retrieval by name.
        - Return the correct algorithm for each set.
        - Return all unique solutions across all sets.
    """
    sol1 = Solution(id="s1", selected_ids={"1"}, objectives={"satisfaction": 10, "effort": 2})
    front1 = ParetoFront([sol1])
    set1 = front1.to_solution_set(name="Run 1", algorithm="NSGA2")

    sol2 = Solution(id="s2", selected_ids={"2"}, objectives={"satisfaction": 20, "effort": 5})
    front2 = ParetoFront([sol2])
    set2 = front2.to_solution_set(name="Run 2", algorithm="MOEA/D")

    collection = SolutionSetCollection(name="Test Collection", sets=[set1, set2])

    assert len(collection) == 2
    assert collection.get_by_name("Run 1").name == "Run 1"
    assert collection.get_by_name("Run 2").algorithm == "MOEA/D"
    assert len(collection.get_all_unique_solutions()) == 2