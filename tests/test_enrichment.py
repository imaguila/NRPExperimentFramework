# tests/test_enrichment.py
"""
Unit tests for the NRP enrichment module.

These tests verify that the NRPEnricher correctly computes various
enrichment indicators such as productivity, effectiveness, squandering,
dirtiness, annoyance, and scope.
"""

import pytest
import pandas as pd
from src.domain.plugins.nrp_enrichment import NRPEnricher


def test_compute_indicators_basic():
    """
    Verify that basic indicators are computed correctly.

    Tests: productivity, effectiveness, squandering, dirtiness, annoyance.
    """
    df = pd.DataFrame({
        "satisfaction": [10, 20, 30],
        "effort": [2, 4, 5],
        "cost": [5, 10, 15],
        "dissatisfaction": [1, 2, 3],
        "prevalence": [3, 6, 9],
        "instability": [2, 3, 4],
        "time": [1, 2, 3],
    })

    indicators = ["productivity", "effectiveness", "squandering", "dirtiness", "annoyance"]
    result = NRPEnricher.compute_indicators(df, indicators)

    # productivity = satisfaction / effort
    assert result["productivity"].tolist() == [10/2, 20/4, 30/5]  # [5.0, 5.0, 6.0]

    # effectiveness = satisfaction / cost
    assert result["effectiveness"].tolist() == [10/5, 20/10, 30/15]  # [2.0, 2.0, 2.0]

    # squandering = (effort_max - effort) / effort_max  (effort_max = 5)
    assert result["squandering"].tolist() == [(5-2)/5, (5-4)/5, (5-5)/5]  # [0.6, 0.2, 0.0]

    # dirtiness = dissatisfaction / effort (with EPS protection)
    assert result["dirtiness"].tolist() == [1/2, 2/4, 3/5]  # [0.5, 0.5, 0.6]

    # annoyance = dissatisfaction / satisfaction
    assert result["annoyance"].tolist() == [1/10, 2/20, 3/30]  # [0.1, 0.1, 0.1]


def test_compute_scope_indicator():
    """
    Verify the 'scope' indicator (ratio of included requirements).

    The scope is computed as (number of selected requirements) / (total requirements).
    """
    df = pd.DataFrame({
        "req_R1": [1, 0, 1],
        "req_R2": [1, 1, 0],
        "req_R3": [0, 1, 1],
    })
    result = NRPEnricher.compute_indicators(df, ["scope"], decision_var_prefix="req_")

    # scope = (number of included reqs) / total reqs (3)
    expected_scope = [2/3, 2/3, 2/3]  # row1: R1+R2=2, row2: R2+R3=2, row3: R1+R3=2
    assert result["scope"].tolist() == pytest.approx(expected_scope, abs=1e-6)


def test_compute_indicators_missing_columns():
    """
    Verify that indicators requiring missing columns are not computed.

    'effectiveness' requires 'cost', which is not present in the DataFrame,
    so it should be omitted from the result.
    """
    df = pd.DataFrame({"satisfaction": [10, 20], "effort": [2, 4]})
    indicators = ["productivity", "effectiveness"]  # effectiveness requires 'cost'
    result = NRPEnricher.compute_indicators(df, indicators)

    assert "productivity" in result.columns
    assert "effectiveness" not in result.columns  # not computed because 'cost' is missing