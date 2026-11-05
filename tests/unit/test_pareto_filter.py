# Location: tests/test_pareto_filter.py
"""
Tests for Pareto front non‑dominance filtering.

These tests verify that the `filter_non_dominated` method correctly
identifies and retains only the non‑dominated solutions from a Pareto front.
"""

import pytest
from src.domain.core.solution import Solution
from src.domain.core.paretofront import ParetoFront


def test_filter_non_dominated():
    """
    Test that non‑dominated filtering correctly removes dominated solutions.

    Given four solutions:
        - s1: (cost=10, quality=80)  → dominated by s2
        - s2: (cost=8,  quality=90)  → non‑dominated
        - s3: (cost=12, quality=95)  → non‑dominated (better quality than s2)
        - s4: (cost=7,  quality=70)  → non‑dominated (better cost than s2)

    With objective directions: cost minimised, quality maximised.
    The filtered front should contain s2, s3, and s4.
    """
    sol1 = Solution(id="s1", selected_ids={"1"}, objectives={"cost": 10, "quality": 80})  # Dominated by s2
    sol2 = Solution(id="s2", selected_ids={"2"}, objectives={"cost": 8, "quality": 90})   # Non‑dominated
    sol3 = Solution(id="s3", selected_ids={"3"}, objectives={"cost": 12, "quality": 95})  # Non‑dominated (better quality than s2)
    sol4 = Solution(id="s4", selected_ids={"4"}, objectives={"cost": 7, "quality": 70})   # Non‑dominated (better cost than s2)

    front = ParetoFront([sol1, sol2, sol3, sol4])
    directions = {"cost": "min", "quality": "max"}

    filtered = front.filter_non_dominated(directions)

    # The filtered front should contain s2, s3, and s4 (3 solutions)
    assert len(filtered) == 3
    assert sol2.id in [s.id for s in filtered.solutions]
    assert sol3.id in [s.id for s in filtered.solutions]
    assert sol4.id in [s.id for s in filtered.solutions]
    assert sol1.id not in [s.id for s in filtered.solutions]