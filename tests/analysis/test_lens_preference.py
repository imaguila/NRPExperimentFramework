# tests/test_lens_preference.py
"""
Unit tests for the Preference lens (MCDM).

These tests verify that the Preference lens correctly implements
multi‑criteria decision‑making methods: Weighted Sum, TOPSIS, VIKOR,
and Reference Point. They also check the top‑N filter and schema generation.
"""

import pytest
import pandas as pd
from src.analysis.lenses.preference import PreferenceLens


@pytest.fixture
def preference_df():
    """Provide a sample DataFrame with solutions and criteria."""
    return pd.DataFrame({
        "id": ["s1", "s2", "s3", "s4"],
        "cost": [10, 20, 15, 25],
        "quality": [80, 90, 85, 70],
        "risk": [2, 5, 3, 8],
    })


def test_preference_lens_weighted_sum(preference_df):
    """
    Test the Weighted Sum method.

    The best solution according to the weighted sum should be s1.
    """
    lens = PreferenceLens()
    params = {
        "method": "Weighted Sum",
        "maximize": ["quality"],
        "minimize": ["cost", "risk"],
        "top_n": 4,
    }
    result = lens.apply(preference_df, params)
    assert "preference_score" in result.columns
    assert "preference_rank" in result.columns
    # s1 is the best solution according to Weighted Sum
    assert result.iloc[0]["id"] == "s1"


def test_preference_lens_topsis(preference_df):
    """
    Test the TOPSIS method.

    TOPSIS scores should be between 0 and 1.
    """
    lens = PreferenceLens()
    params = {
        "method": "TOPSIS",
        "maximize": ["quality"],
        "minimize": ["cost", "risk"],
        "top_n": 4,
    }
    result = lens.apply(preference_df, params)
    assert "preference_score" in result.columns
    assert all(0 <= x <= 1 for x in result["preference_score"])


def test_preference_lens_vikor(preference_df):
    """
    Test the VIKOR method.

    VIKOR scores should be between 0 and 1.
    """
    lens = PreferenceLens()
    params = {
        "method": "VIKOR",
        "maximize": ["quality"],
        "minimize": ["cost", "risk"],
        "top_n": 4,
    }
    result = lens.apply(preference_df, params)
    assert "preference_score" in result.columns
    assert all(0 <= x <= 1 for x in result["preference_score"])


def test_preference_lens_reference_point(preference_df):
    """
    Test the Reference Point method.

    Either s1 or s2 should be among the top solutions.
    """
    lens = PreferenceLens()
    params = {
        "method": "Reference Point",
        "maximize": ["quality"],
        "minimize": ["cost", "risk"],
        "top_n": 4,
    }
    result = lens.apply(preference_df, params)
    assert "preference_score" in result.columns
    top_ids = result["id"].tolist()
    assert "s1" in top_ids or "s2" in top_ids


def test_preference_lens_top_n_filter(preference_df):
    """
    Test that the top‑N parameter correctly limits the number of results.
    """
    lens = PreferenceLens()
    params = {
        "method": "Weighted Sum",
        "maximize": ["quality"],
        "minimize": ["cost"],
        "top_n": 2,
    }
    result = lens.apply(preference_df, params)
    assert len(result) == 2


def test_preference_lens_no_criteria(preference_df):
    """
    Test that the lens returns the original DataFrame when no criteria are specified.
    """
    lens = PreferenceLens()
    params = {"method": "Weighted Sum", "maximize": [], "minimize": []}
    result = lens.apply(preference_df, params)
    pd.testing.assert_frame_equal(result, preference_df)


def test_preference_lens_schema(preference_df):
    """
    Test that the lens schema contains the expected controls.
    """
    lens = PreferenceLens()
    schema = lens.get_schema(
        preference_df,
        ["cost", "quality", "risk"]
    )
    assert isinstance(schema, list)
    keys = [item["key"] for item in schema]
    assert "method" in keys
    assert "maximize" in keys
    assert "minimize" in keys
    assert "top_n" in keys