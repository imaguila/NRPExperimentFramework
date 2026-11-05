"""Tests for src.solving.core.metrics.ParetoMetrics.

These tests deliberately use lightweight front and solution objects. This
keeps the metric tests independent from constructors, CSV import, and other
domain behaviour while preserving the public attributes consumed by
ParetoMetrics: front.solutions, front.objective_directions,
solution.objectives, and solution.is_feasible.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from src.solving.core.metrics import ParetoMetrics


@dataclass
class DummySolution:
    """Minimal solution object required by ParetoMetrics."""

    objectives: dict
    is_feasible: bool = True
    id: str = "solution"


class DummyFront:
    """Minimal Pareto-front object required by ParetoMetrics."""

    def __init__(self, objective_rows, directions, feasible=None):
        if feasible is None:
            feasible = [True] * len(objective_rows)
        self.solutions = [
            DummySolution(
                objectives=dict(row),
                is_feasible=is_feasible,
                id=f"s{index}",
            )
            for index, (row, is_feasible) in enumerate(
                zip(objective_rows, feasible), start=1
            )
        ]
        self.objective_directions = dict(directions)


# ---------------------------------------------------------------------------
# Objective extraction and direction handling
# ---------------------------------------------------------------------------


def test_get_solution_vectors_transforms_maximisation_to_minimisation():
    front = DummyFront(
        [
            {"satisfaction": 9.0, "effort": 4.0},
            {"satisfaction": 7.0, "effort": 2.0},
        ],
        {"satisfaction": "max", "effort": "min"},
    )

    vectors = ParetoMetrics.get_solution_vectors(
        front,
        objective_names=["satisfaction", "effort"],
    )

    np.testing.assert_allclose(vectors, [[-9.0, 4.0], [-7.0, 2.0]])


def test_explicit_directions_override_front_metadata():
    front = DummyFront(
        [{"a": 3.0, "b": 5.0}],
        {"a": "min", "b": "min"},
    )

    vectors = ParetoMetrics.get_solution_vectors(
        front,
        objective_names=["a", "b"],
        directions={"a": "max", "b": "min"},
    )

    np.testing.assert_allclose(vectors, [[-3.0, 5.0]])


def test_infeasible_solutions_are_excluded_from_vectors_by_default():
    front = DummyFront(
        [{"x": 1.0}, {"x": 2.0}],
        {"x": "min"},
        feasible=[True, False],
    )

    vectors = ParetoMetrics.get_solution_vectors(front)

    np.testing.assert_allclose(vectors, [[1.0]])


def test_missing_objective_raises_value_error():
    front = DummyFront(
        [{"x": 1.0, "y": 2.0}, {"x": 2.0}],
        {"x": "min", "y": "min"},
    )

    with pytest.raises(ValueError, match="missing"):
        ParetoMetrics.get_solution_vectors(front, ["x", "y"])


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_non_finite_objective_raises_value_error(invalid_value):
    front = DummyFront(
        [{"x": invalid_value}],
        {"x": "min"},
    )

    with pytest.raises(ValueError, match="finite"):
        ParetoMetrics.get_solution_vectors(front)


def test_missing_direction_raises_when_direction_mapping_is_explicit():
    front = DummyFront(
        [{"x": 1.0, "y": 2.0}],
        {"x": "min", "y": "min"},
    )

    with pytest.raises(ValueError, match="direction"):
        ParetoMetrics.get_solution_vectors(
            front,
            objective_names=["x", "y"],
            directions={"x": "min"},
        )


# ---------------------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------------------


def test_hypervolume_empty_front_is_zero():
    front = DummyFront([], {"x": "min", "y": "min"})

    assert ParetoMetrics.hypervolume(front, [5.0, 5.0]) == 0.0


def test_hypervolume_known_one_dimensional_case():
    front = DummyFront(
        [{"x": 2.0}, {"x": 4.0}],
        {"x": "min"},
    )

    assert ParetoMetrics.hypervolume(front, [5.0]) == pytest.approx(3.0)


def test_hypervolume_known_two_dimensional_case():
    # Dominated area with reference (5, 5):
    # (5-1)*(5-4) + (5-2)*(4-2) + (5-4)*(2-1) = 4+6+1 = 11.
    front = DummyFront(
        [
            {"x": 1.0, "y": 4.0},
            {"x": 2.0, "y": 2.0},
            {"x": 4.0, "y": 1.0},
        ],
        {"x": "min", "y": "min"},
    )

    assert ParetoMetrics.hypervolume(front, [5.0, 5.0]) == pytest.approx(11.0)


def test_hypervolume_known_three_dimensional_single_box():
    # One point dominates a box with side lengths 4, 4, and 4.
    front = DummyFront(
        [{"x": 1.0, "y": 2.0, "z": 3.0}],
        {"x": "min", "y": "min", "z": "min"},
    )

    assert ParetoMetrics.hypervolume(
        front, [5.0, 6.0, 7.0]
    ) == pytest.approx(64.0)


def test_hypervolume_mixed_directions_matches_explicit_transformation():
    mixed_front = DummyFront(
        [
            {"value": 9.0, "cost": 4.0},
            {"value": 8.0, "cost": 2.0},
            {"value": 6.0, "cost": 1.0},
        ],
        {"value": "max", "cost": "min"},
    )
    minimisation_front = DummyFront(
        [
            {"negative_value": -9.0, "cost": 4.0},
            {"negative_value": -8.0, "cost": 2.0},
            {"negative_value": -6.0, "cost": 1.0},
        ],
        {"negative_value": "min", "cost": "min"},
    )

    mixed_hv = ParetoMetrics.hypervolume(
        mixed_front,
        reference_point=[5.0, 5.0],
        objective_names=["value", "cost"],
    )
    transformed_hv = ParetoMetrics.hypervolume(
        minimisation_front,
        reference_point=[-5.0, 5.0],
        objective_names=["negative_value", "cost"],
    )

    assert mixed_hv == pytest.approx(transformed_hv)


def test_duplicate_points_do_not_change_hypervolume():
    unique = DummyFront(
        [{"x": 1.0, "y": 4.0}, {"x": 2.0, "y": 2.0}],
        {"x": "min", "y": "min"},
    )
    duplicated = DummyFront(
        [
            {"x": 1.0, "y": 4.0},
            {"x": 2.0, "y": 2.0},
            {"x": 2.0, "y": 2.0},
        ],
        {"x": "min", "y": "min"},
    )

    assert ParetoMetrics.hypervolume(
        unique, [5.0, 5.0]
    ) == pytest.approx(ParetoMetrics.hypervolume(duplicated, [5.0, 5.0]))


def test_dominated_points_do_not_change_hypervolume():
    non_dominated = DummyFront(
        [{"x": 1.0, "y": 4.0}, {"x": 2.0, "y": 2.0}],
        {"x": "min", "y": "min"},
    )
    with_dominated = DummyFront(
        [
            {"x": 1.0, "y": 4.0},
            {"x": 2.0, "y": 2.0},
            {"x": 4.0, "y": 4.0},
        ],
        {"x": "min", "y": "min"},
    )

    assert ParetoMetrics.hypervolume(
        non_dominated, [5.0, 5.0]
    ) == pytest.approx(ParetoMetrics.hypervolume(with_dominated, [5.0, 5.0]))


def test_infeasible_point_does_not_change_hypervolume():
    front = DummyFront(
        [
            {"x": 2.0, "y": 2.0},
            {"x": 1.0, "y": 1.0},
        ],
        {"x": "min", "y": "min"},
        feasible=[True, False],
    )

    assert ParetoMetrics.hypervolume(front, [5.0, 5.0]) == pytest.approx(9.0)


def test_reference_point_dimensionality_is_validated():
    front = DummyFront(
        [{"x": 1.0, "y": 2.0}],
        {"x": "min", "y": "min"},
    )

    with pytest.raises(ValueError, match="dimensionality"):
        ParetoMetrics.hypervolume(front, [5.0])


def test_reference_point_worse_than_a_solution_is_rejected():
    front = DummyFront(
        [{"x": 4.0, "y": 1.0}],
        {"x": "min", "y": "min"},
    )

    with pytest.raises(ValueError, match="Reference point"):
        ParetoMetrics.hypervolume(front, [3.0, 5.0])


def test_point_on_reference_boundary_has_zero_hypervolume():
    front = DummyFront(
        [{"x": 5.0, "y": 2.0}],
        {"x": "min", "y": "min"},
    )

    assert ParetoMetrics.hypervolume(front, [5.0, 5.0]) == 0.0


def test_automatic_reference_is_only_front_local_but_returns_positive_value():
    front = DummyFront(
        [{"x": 1.0, "y": 4.0}, {"x": 2.0, "y": 2.0}],
        {"x": "min", "y": "min"},
    )

    assert ParetoMetrics.hypervolume(front) > 0.0


# ---------------------------------------------------------------------------
# Spacing, extent, compatibility alias, and aggregate API
# ---------------------------------------------------------------------------


def test_spacing_is_zero_for_equally_spaced_points():
    front = DummyFront(
        [{"x": 0.0}, {"x": 1.0}, {"x": 2.0}],
        {"x": "min"},
    )

    assert ParetoMetrics.spacing(front) == pytest.approx(0.0)


def test_spacing_ignores_duplicate_objective_vectors():
    unique = DummyFront(
        [{"x": 0.0}, {"x": 1.0}, {"x": 3.0}],
        {"x": "min"},
    )
    duplicated = DummyFront(
        [{"x": 0.0}, {"x": 1.0}, {"x": 1.0}, {"x": 3.0}],
        {"x": "min"},
    )

    assert ParetoMetrics.spacing(unique) == pytest.approx(
        ParetoMetrics.spacing(duplicated)
    )


def test_objective_space_extent_is_bounding_box_diagonal():
    front = DummyFront(
        [{"x": 1.0, "y": 2.0}, {"x": 4.0, "y": 6.0}],
        {"x": "min", "y": "min"},
    )

    assert ParetoMetrics.objective_space_extent(front) == pytest.approx(5.0)


def test_spread_is_backward_compatible_extent_alias():
    front = DummyFront(
        [{"x": 1.0, "y": 2.0}, {"x": 4.0, "y": 6.0}],
        {"x": "min", "y": "min"},
    )

    assert ParetoMetrics.spread(front) == pytest.approx(
        ParetoMetrics.objective_space_extent(front)
    )


def test_compute_all_metrics_returns_expected_keys_and_distinct_count():
    front = DummyFront(
        [
            {"x": 1.0, "y": 4.0},
            {"x": 2.0, "y": 2.0},
            {"x": 2.0, "y": 2.0},
        ],
        {"x": "min", "y": "min"},
    )

    metrics = ParetoMetrics.compute_all_metrics(
        front,
        reference_point=[5.0, 5.0],
    )

    assert metrics["Solutions"] == 2
    assert metrics["Hypervolume"] == pytest.approx(10.0)
    assert metrics["Spacing"] >= 0.0
    assert metrics["Objective-space extent"] == pytest.approx(np.sqrt(5.0))
    assert metrics["Spread"] == pytest.approx(metrics["Objective-space extent"])


def test_compute_all_metrics_empty_front_returns_empty_dictionary():
    front = DummyFront([], {"x": "min"})

    assert ParetoMetrics.compute_all_metrics(front) == {}
