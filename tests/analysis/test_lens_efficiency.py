# tests/test_lens_efficiency.py
"""
Unit tests for the Efficiency lens.

These tests verify that the Efficiency lens correctly computes various
benefit‑cost trade‑off metrics (Benefit/Cost Ratio, Normalized Ratio)
and applies the top‑N filter correctly.
"""

import pytest
import pandas as pd
from src.analysis.lenses.efficiency import EfficiencyLens


def test_efficiency_lens_benefit_cost_ratio():
    """
    Test the Benefit/Cost Ratio calculation and ordering.

    Ratios: s1=5, s2=5, s3=6. Sorted descending: s3, s1, s2.
    """
    df = pd.DataFrame({
        "id": ["s1", "s2", "s3"],
        "benefit": [10, 20, 30],
        "cost": [2, 4, 5]
    })
    lens = EfficiencyLens()
    params = {
        "method": "Benefit/Cost Ratio",
        "benefit": "benefit",
        "cost": "cost",
        "top_n": 3
    }
    result = lens.apply(df, params)

    # Expected order: s3 (6.0), s1 (5.0), s2 (5.0)
    assert result["id"].tolist() == ["s3", "s1", "s2"]
    assert result["efficiency_score"].tolist() == [6.0, 5.0, 5.0]


def test_efficiency_lens_normalized_ratio():
    """
    Test the Normalized Ratio calculation (min‑max normalisation).

    benefit_norm: [0, 0.5, 1]; cost_norm: [0, 0.666, 1]
    Normalised ratio = benefit_norm / (cost_norm + EPS)
    Expected: s3≈1.0, s2≈0.75, s1≈0.0
    """
    df = pd.DataFrame({
        "id": ["s1", "s2", "s3"],
        "benefit": [10, 20, 30],
        "cost": [2, 4, 5]
    })
    lens = EfficiencyLens()
    params = {
        "method": "Normalized Ratio",
        "benefit": "benefit",
        "cost": "cost",
        "top_n": 3
    }
    result = lens.apply(df, params)

    # s3 should be closest to 1.0, s2 around 0.75, s1 at 0.0
    assert result["efficiency_score"].iloc[0] == pytest.approx(1.0, abs=1e-6)  # s3
    assert result["efficiency_score"].iloc[1] == pytest.approx(0.75, abs=1e-2)  # s2
    assert result["efficiency_score"].iloc[2] == pytest.approx(0.0, abs=1e-6)  # s1


def test_efficiency_lens_top_n_filter():
    """
    Test that the top‑N parameter correctly limits the number of results.
    """
    df = pd.DataFrame({
        "id": [f"s{i}" for i in range(10)],
        "benefit": list(range(10, 20)),
        "cost": [1] * 10
    })
    lens = EfficiencyLens()
    params = {
        "method": "Benefit/Cost Ratio",
        "benefit": "benefit",
        "cost": "cost",
        "top_n": 3
    }
    result = lens.apply(df, params)
    assert len(result) == 3
    # The top 3 should be the ones with highest benefit (s9, s8, s7)
    assert result["id"].tolist() == ["s9", "s8", "s7"]