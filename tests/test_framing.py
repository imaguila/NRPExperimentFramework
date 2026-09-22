# tests/test_framing.py
"""
Unit tests for the framing module.

These tests verify the functionality of framing dimension detection,
bound computation, and the application of bounds to filter DataFrames.
"""

import pytest
import pandas as pd
from src.analysis.framing import get_framing_dimensions, get_dimension_bounds, apply_framing_bounds


def test_get_framing_dimensions_with_metrics_and_indicators():
    """
    Verify that metrics and indicators are combined without duplicates.
    """
    metrics = ["cost", "effort"]
    indicators = ["scope", "productivity"]
    dims = get_framing_dimensions(metrics, indicators)
    assert dims == ["cost", "effort", "scope", "productivity"]


def test_get_framing_dimensions_auto_infer():
    """
    Verify that numeric dimensions are automatically inferred from the DataFrame
    when no explicit lists are provided.

    Columns starting with 'req_' and non-numeric columns should be excluded.
    """
    df = pd.DataFrame({
        "id": ["a", "b"],
        "cost": [10, 20],
        "effort": [5, 15],
        "req_R1": [1, 0],  # excluded because it starts with 'req_'
        "name": ["x", "y"]  # excluded because it is non-numeric
    })
    dims = get_framing_dimensions(df=df)
    assert set(dims) == {"cost", "effort"}


def test_get_dimension_bounds():
    """
    Verify that min and max bounds are correctly computed for numeric columns.
    """
    df = pd.DataFrame({"cost": [10, 20, 30], "effort": [5, 10, 15]})
    bounds = get_dimension_bounds(df, ["cost", "effort"])
    assert bounds == {"cost": (10.0, 30.0), "effort": (5.0, 15.0)}


def test_apply_framing_bounds():
    """
    Verify that rows outside the specified bounds are filtered out.

    Only rows that satisfy all bound conditions should remain.
    """
    df = pd.DataFrame({"cost": [10, 20, 30, 40], "effort": [5, 10, 15, 20]})
    bounds = {"cost": (15.0, 35.0), "effort": (8.0, 18.0)}
    filtered = apply_framing_bounds(df, bounds)

    # Only rows that satisfy both conditions should remain:
    # Row 0: cost=10 → excluded
    # Row 1: cost=20, effort=10 → included
    # Row 2: cost=30, effort=15 → included
    # Row 3: cost=40 → excluded
    assert len(filtered) == 2
    assert filtered["cost"].tolist() == [20, 30]
    assert filtered["effort"].tolist() == [10, 15]


def test_apply_framing_bounds_empty_bounds():
    """
    Verify that an empty bounds dictionary returns the original DataFrame unchanged.
    """
    df = pd.DataFrame({"x": [1, 2, 3]})
    filtered = apply_framing_bounds(df, {})
    pd.testing.assert_frame_equal(filtered, df)