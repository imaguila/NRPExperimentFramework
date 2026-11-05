# tests/test_lens_manual.py
"""
Unit tests for the Manual Selection lens.

These tests verify that the Manual lens correctly filters a DataFrame
to keep only solutions with the specified IDs, handling edge cases such
as empty selections, non‑existent IDs, missing ID columns, and empty DataFrames.
"""

import pytest
import pandas as pd
from src.analysis.lenses.manual import ManualLens


@pytest.fixture
def manual_df():
    """Provide a sample DataFrame with IDs and numeric columns."""
    return pd.DataFrame({
        "id": ["s1", "s2", "s3", "s4", "s5"],
        "cost": [10, 20, 15, 25, 30],
        "quality": [80, 90, 85, 70, 95],
    })


def test_manual_lens_basic(manual_df):
    """
    Test that the Manual lens filters by ID correctly.

    Only the solutions with IDs "s2" and "s4" should be retained.
    """
    lens = ManualLens()
    params = {
        "selected_ids": ["s2", "s4"],
    }
    result = lens.apply(manual_df, params)

    assert len(result) == 2
    assert set(result["id"].tolist()) == {"s2", "s4"}


def test_manual_lens_no_ids(manual_df):
    """
    Test that an empty list of IDs returns an empty DataFrame.
    """
    lens = ManualLens()
    params = {
        "selected_ids": [],
    }
    result = lens.apply(manual_df, params)
    assert result.empty


def test_manual_lens_ids_not_present(manual_df):
    """
    Test that non‑existent IDs cause no error and return an empty DataFrame.
    """
    lens = ManualLens()
    params = {
        "selected_ids": ["s99", "s100"],
    }
    result = lens.apply(manual_df, params)
    assert result.empty


def test_manual_lens_no_id_column():
    """
    Test that a DataFrame without an 'id' column returns an empty DataFrame.
    """
    df = pd.DataFrame({"x": [1, 2, 3]})
    lens = ManualLens()
    params = {"selected_ids": ["1"]}
    result = lens.apply(df, params)
    assert result.empty


def test_manual_lens_empty_dataframe():
    """
    Test that an empty DataFrame returns an empty DataFrame.
    """
    df = pd.DataFrame()
    lens = ManualLens()
    params = {"selected_ids": ["s1"]}
    result = lens.apply(df, params)
    assert result.empty


def test_manual_lens_schema():
    """
    Test that the Manual lens schema is empty (no configurable parameters).
    """
    schema = ManualLens.get_schema(pd.DataFrame(), [])
    assert schema == []