# tests/test_lens_consensus.py
"""
Unit tests for the Consensus lens (SOI voting).

These tests verify that the Consensus lens correctly aggregates multiple
Sets of Interest (SOIs) using threshold‑based voting, and that the resulting
DataFrame contains the expected consensus scores and group labels.
"""

import pytest
import pandas as pd
from unittest.mock import Mock

from src.analysis.lenses.consensus import ConsensusLens


@pytest.fixture
def consensus_df():
    """Provide a sample DataFrame with solutions for consensus testing."""
    return pd.DataFrame({
        "id": ["s1", "s2", "s3", "s4", "s5"],
        "cost": [10, 20, 15, 25, 30],
        "quality": [80, 90, 85, 70, 95],
    })


@pytest.fixture
def mock_sois():
    """Create three mock SOIs with overlapping solution IDs."""
    soi1 = Mock()
    soi1.name = "SOI_A"
    soi1.solution_ids = ["s1", "s2", "s3"]

    soi2 = Mock()
    soi2.name = "SOI_B"
    soi2.solution_ids = ["s2", "s3", "s4"]

    soi3 = Mock()
    soi3.name = "SOI_C"
    soi3.solution_ids = ["s3", "s4", "s5"]
    return [soi1, soi2, soi3]


def test_consensus_lens_basic(consensus_df, mock_sois):
    """
    Test that the Consensus lens correctly computes support scores and filters
    by threshold.

    With 3 SOIs and threshold 0.5, solutions present in at least 2 SOIs
    (s2, s3, s4) should be retained.
    """
    lens = ConsensusLens()
    params = {
        "selected_sois": ["SOI_A", "SOI_B", "SOI_C"],
        "threshold": 0.5,
    }
    context = {"saved_sois": mock_sois}
    result = lens.apply(consensus_df, params, context=context)

    assert "consensus_score" in result.columns
    assert "group_label" in result.columns
    assert len(result) == 3
    assert set(result["id"].tolist()) == {"s2", "s3", "s4"}


def test_consensus_lens_threshold_filter(consensus_df, mock_sois):
    """
    Test that increasing the threshold filters out solutions with lower support.

    With threshold 0.7, only solutions present in all 3 SOIs (s3) should remain.
    """
    lens = ConsensusLens()
    params = {
        "selected_sois": ["SOI_A", "SOI_B", "SOI_C"],
        "threshold": 0.7,
    }
    context = {"saved_sois": mock_sois}
    result = lens.apply(consensus_df, params, context=context)
    assert len(result) == 1
    assert result.iloc[0]["id"] == "s3"


def test_consensus_lens_less_than_two_sois(consensus_df, mock_sois):
    """
    Test that the lens returns an empty DataFrame when fewer than 2 SOIs are selected.
    """
    lens = ConsensusLens()
    params = {
        "selected_sois": ["SOI_A"],
        "threshold": 0.5,
    }
    context = {"saved_sois": mock_sois}
    result = lens.apply(consensus_df, params, context=context)
    assert result.empty


def test_consensus_lens_no_selected_sois(consensus_df, mock_sois):
    """
    Test that the original DataFrame is returned unchanged when no SOIs are selected.
    """
    lens = ConsensusLens()
    params = {"selected_sois": [], "threshold": 0.5}
    context = {"saved_sois": mock_sois}
    result = lens.apply(consensus_df, params, context=context)
    pd.testing.assert_frame_equal(result, consensus_df)


def test_consensus_lens_no_context(consensus_df):
    """
    Test that the lens returns an empty DataFrame when no context (SOIs) is provided.
    """
    lens = ConsensusLens()
    params = {"selected_sois": ["SOI_A"], "threshold": 0.5}
    result = lens.apply(consensus_df, params, context=None)
    assert result.empty


def test_consensus_lens_schema(consensus_df, mock_sois):
    """
    Test that the lens schema includes the expected controls: SOI selection
    and threshold slider.
    """
    lens = ConsensusLens()
    schema = lens.get_schema(
        consensus_df,
        [],
        context={"saved_sois": mock_sois}
    )
    assert isinstance(schema, list)
    keys = [item["key"] for item in schema]
    assert "selected_sois" in keys
    assert "threshold" in keys