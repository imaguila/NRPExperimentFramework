# Location: tests/test_dominance.py
"""
Tests for dominance between solutions.

These tests verify the correctness of the dominance relationship logic,
including basic dominance, incomparable solutions, and constrained dominance
(feasible vs infeasible solutions).
"""

import pytest
from src.domain.core.solution import Solution


def test_dominance_basic():
    """
    Test that a solution that is better in all objectives dominates another.

    Solution A (cost=10, quality=90) dominates Solution B (cost=20, quality=80)
    because it is better in both objectives.
    """
    sol_a = Solution(
        id="A",
        selected_ids={"1"},
        objectives={"cost": 10, "quality": 90}
    )
    sol_b = Solution(
        id="B",
        selected_ids={"2"},
        objectives={"cost": 20, "quality": 80}
    )

    assert sol_a.dominates(sol_b, {"cost": "min", "quality": "max"}) is True
    assert sol_b.dominates(sol_a, {"cost": "min", "quality": "max"}) is False


def test_dominance_incomparable():
    """
    Test that two solutions are incomparable when one is better in one objective
    but worse in another.

    Solution A is better in cost (10 vs 20) but worse in quality (80 vs 95).
    Solution B is better in quality (95 vs 80) but worse in cost (20 vs 10).
    Neither dominates the other.
    """
    sol_a = Solution(
        id="A",
        selected_ids={"1"},
        objectives={"cost": 10, "quality": 80}
    )
    sol_b = Solution(
        id="B",
        selected_ids={"2"},
        objectives={"cost": 20, "quality": 95}
    )

    assert sol_a.dominates(sol_b, {"cost": "min", "quality": "max"}) is False
    assert sol_b.dominates(sol_a, {"cost": "min", "quality": "max"}) is False


def test_dominance_with_constraints():
    """
    Test constrained dominance: a feasible solution dominates an infeasible one.

    Feasible solution A (cost=10) dominates infeasible solution B (cost=5)
    because feasibility is considered before objective values.
    """
    sol_feasible = Solution(
        id="A",
        selected_ids={"1"},
        objectives={"cost": 10},
        is_feasible=True
    )
    sol_infeasible = Solution(
        id="B",
        selected_ids={"2"},
        objectives={"cost": 5},
        is_feasible=False
    )

    assert sol_feasible.dominates(sol_infeasible, {"cost": "min"}) is True
    assert sol_infeasible.dominates(sol_feasible, {"cost": "min"}) is False