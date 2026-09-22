# tests/test_stakeholder_coverage.py
"""
Unit tests for NRP stakeholder coverage calculation.

These tests verify that the stakeholder coverage functions correctly extract
original satisfaction values, build the stakeholder‑requirement request matrix,
and compute coverage scores for solutions.
"""

import pytest
import pandas as pd
from src.domain.core.model import OptimizationModel
from src.domain.core.item import DecisionItem
from src.domain.core.definitions import Evaluator
from src.domain.plugins.nrp import NRPPlugin
from src.domain.plugins.nrp_coverage import _get_original_satisfaction


@pytest.fixture
def model_with_stakeholders():
    """
    Create a synthetic model with:
        - 3 requirements (R1, R2, R3)
        - 2 stakeholders (S1, S2) in evaluators
        - Multivalued satisfaction values:
            R1: [3, 5]  (S1=3, S2=5)
            R2: [0, 2]  (S1=0, S2=2)
            R3: [4, 1]  (S1=4, S2=1)
    """
    items = {
        "R1": DecisionItem("R1", "Req 1", {"satisfaction": [3, 5]}),
        "R2": DecisionItem("R2", "Req 2", {"satisfaction": [0, 2]}),
        "R3": DecisionItem("R3", "Req 3", {"satisfaction": [4, 1]}),
    }

    raw_data = {
        "requirements": [
            {"id": "R1", "attributes": {"satisfaction": [3, 5]}},
            {"id": "R2", "attributes": {"satisfaction": [0, 2]}},
            {"id": "R3", "attributes": {"satisfaction": [4, 1]}},
        ],
        "stakeholders": [
            {"id": "S1", "weight": 1.0},
            {"id": "S2", "weight": 1.0},
        ]
    }

    evaluators = {
        "stakeholders": [
            Evaluator(id="S1", weight=1.0, group="stakeholders"),
            Evaluator(id="S2", weight=1.0, group="stakeholders"),
        ]
    }

    model = OptimizationModel(
        name="StakeholderTest",
        items=items,
        raw_data=raw_data,
        evaluators=evaluators,
        plugin=NRPPlugin
    )
    return model


def test_get_original_satisfaction(model_with_stakeholders):
    """
    Verify that `_get_original_satisfaction` correctly extracts the original
    multivalued satisfaction lists from the model.
    """
    sat_by_req = _get_original_satisfaction(model_with_stakeholders)

    assert sat_by_req == {
        "R1": [3.0, 5.0],
        "R2": [0.0, 2.0],
        "R3": [4.0, 1.0]
    }


def test_build_stakeholder_requirement_matrix(model_with_stakeholders):
    """
    Verify that the stakeholder‑requirement request matrix is correctly built.

    The matrix should contain 1 if the stakeholder requested the requirement
    (satisfaction > 0), and 0 otherwise.
    """
    matrix = NRPPlugin.build_stakeholder_requirement_matrix(model_with_stakeholders)

    expected = pd.DataFrame(
        {
            "S1": [1, 0, 1],  # R1 and R3 requested by S1; R2 not
            "S2": [1, 1, 1],  # All requested by S2
        },
        index=["R1", "R2", "R3"]
    )
    pd.testing.assert_frame_equal(matrix, expected)


def test_compute_stakeholder_coverage(model_with_stakeholders):
    """
    Verify that stakeholder coverage is correctly computed for a DataFrame
    of solutions.

    For sol1: R1(3,5) + R2(0,2) = (3,7) → S1=3, S2=7
    For sol2: R2(0,2) + R3(4,1) = (4,3) → S1=4, S2=3
    """
    df = pd.DataFrame({
        "id": ["sol1", "sol2"],
        "req_R1": [1, 0],  # sol1 includes R1, sol2 does not
        "req_R2": [1, 1],  # both include R2
        "req_R3": [0, 1],  # sol1 does not include R3, sol2 does
    })

    result = NRPPlugin.compute_stakeholder_coverage(df, model_with_stakeholders)

    assert result["stcov_S1"].tolist() == [3.0, 4.0]
    assert result["stcov_S2"].tolist() == [7.0, 3.0]