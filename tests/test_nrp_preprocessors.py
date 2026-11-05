# Location: tests/test_nrp_preprocessors.py
"""
Unit tests for NRP preprocessors.

These tests verify the behaviour of the exclusion and coupling preprocessors,
including branching on exclusions, cascade deletion via precedences, and
Disjoint‑Set Union (DSU) fusion of coupled items.
"""

import pytest
from src.domain.core.model import OptimizationModel
from src.domain.core.item import DecisionItem
from src.domain.plugins import nrp_coupling
from src.domain.plugins import nrp_exclusion


@pytest.fixture
def base_model():
    """
    Create a basic model with 3 requirements and composite relationships.

    The model includes:
        - Precedence: R3 depends on R1 (selecting R3 requires R1).
        - Exclusion: R1 and R2 are mutually exclusive.
        - No couplings initially.
    """
    items = {
        "R1": DecisionItem("R1", "Req 1", {"cost": 10.0}),
        "R2": DecisionItem("R2", "Req 2", {"cost": 20.0}),
        "R3": DecisionItem("R3", "Req 3", {"cost": 15.0}),
    }
    relationships = {
        "precedences": [{"req1": "R3", "req2": "R1"}],  # R3 depends on R1 (selecting R3 requires R1)
        "exclusions": [{"req1": "R1", "req2": "R2"}],   # R1 and R2 exclude each other
        "couplings": []
    }
    return OptimizationModel(name="TestModel", items=items, relationships=relationships)


def test_exclusion_cascade_branching(base_model):
    """
    Test that exclusion branching correctly handles transitive dependencies.

    The model has an exclusion between R1 and R2, with R3 depending on R1.
    Expected branches:
        - Branch A: Remove R1 → cascade removes R3 → only R2 remains.
        - Branch B: Remove R2 → R1 and R3 remain.
    """
    branches = nrp_exclusion.run(base_model)

    assert len(branches) == 2

    items_branch_a = set(branches[0].items.keys())
    assert items_branch_a == {"R2"}

    items_branch_b = set(branches[1].items.keys())
    assert items_branch_b == {"R1", "R3"}


def test_coupling_dsu_fusion(base_model):
    """
    Test that the coupling preprocessor fuses coupled items using DSU.

    After adding a coupling between R1 and R2, they should be fused into a
    single item "R1_R2" with aggregated attributes (cost = 10 + 20 = 30).
    """
    # Add a coupling between R1 and R2
    base_model.relationships["couplings"] = [{"req1": "R1", "req2": "R2"}]

    results = nrp_coupling.run(base_model)
    unified_model = results[0]

    # R1 and R2 should be fused into "R1_R2" with the sum of their costs
    assert "R1_R2" in unified_model.items
    assert unified_model.items["R1_R2"].attributes["cost"] == 30.0