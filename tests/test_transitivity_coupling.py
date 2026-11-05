# Location: tests/test_exclusion_transitivity_with_coupling.py
"""
Test: Exclusions with transitivity and couplings.

Verifies that the preprocessor correctly handles the interaction between:
    - Exclusions
    - Transitive precedences (R3 -> R2 -> R1, i.e., R3 requires R2, R2 requires R1)
    - Couplings (R4 <-> R5)
"""

import pytest
from src.domain.core.item import DecisionItem
from src.domain.core.model import OptimizationModel
from src.domain.plugins import nrp_coupling
from src.domain.plugins import nrp_exclusion


@pytest.fixture
def model_with_transitivity_and_coupling():
    """
    Create a model with:
        - R3 requires R2, R2 requires R1 (R3 -> R2 -> R1)
        - R4 and R5 coupled
        - Exclusion between R1 and R4
    """
    items = {
        "R1": DecisionItem("R1", "Req 1", {"cost": 10.0}),
        "R2": DecisionItem("R2", "Req 2", {"cost": 20.0}),
        "R3": DecisionItem("R3", "Req 3", {"cost": 15.0}),
        "R4": DecisionItem("R4", "Req 4", {"cost": 30.0}),
        "R5": DecisionItem("R5", "Req 5", {"cost": 25.0}),
    }
    relationships = {
        "precedences": [
            {"from": "R2", "to": "R1"},  # R2 depends on R1 (selecting R2 requires R1)
            {"from": "R3", "to": "R2"}   # R3 depends on R2 (selecting R3 requires R2)
        ],
        "couplings": [
            {"req1": "R4", "req2": "R5"}  # R4 and R5 are coupled
        ],
        "exclusions": [
            {"req1": "R1", "req2": "R4"}  # R1 and R4 exclude each other
        ],
        "value_dependencies": []
    }
    return OptimizationModel(
        name="TransitivityCouplingTest",
        items=items,
        relationships=relationships
    )


def test_exclusion_transitivity_and_coupling(model_with_transitivity_and_coupling):
    """
    Verify that:
        1. Removing R1 also removes R2 and R3 by transitivity.
        2. Removing R4 also removes R5 by coupling.
        3. The coupling R4-R5 is correctly fused in the branch where they survive
           (note: exclusion.run() does NOT apply coupling, only deletion, so
           R4 and R5 remain separate in that branch).
    """
    branches = nrp_exclusion.run(model_with_transitivity_and_coupling)

    assert len(branches) == 2

    # --- Branch A: Remove R1 ---
    branch_excl_r1 = next(b for b in branches if "Excl(R1)" in b.branch_name)

    # R1, R2, R3 must be removed (R1 removed, R2 and R3 by transitivity)
    assert "R1" not in branch_excl_r1.items
    assert "R2" not in branch_excl_r1.items
    assert "R3" not in branch_excl_r1.items

    # R4 and R5 remain as individual items (coupling not yet applied)
    assert "R4" in branch_excl_r1.items
    assert "R5" in branch_excl_r1.items
    assert "R4_R5" not in branch_excl_r1.items  # Not fused yet

    # --- Branch B: Remove R4 ---
    branch_excl_r4 = next(b for b in branches if "Excl(R4)" in b.branch_name)

    # R4 must be removed
    assert "R4" not in branch_excl_r4.items

    # R5 must be removed BY COUPLING (R4 and R5 are coupled)
    assert "R5" not in branch_excl_r4.items

    # R1, R2, R3 must remain
    assert "R1" in branch_excl_r4.items
    assert "R2" in branch_excl_r4.items
    assert "R3" in branch_excl_r4.items


def test_full_pipeline_with_transitivity_and_coupling(model_with_transitivity_and_coupling):
    """
    Test the full preprocessing pipeline (coupling, value dependencies, exclusion)
    on the model with transitivity and couplings.

    The pipeline should first fuse R4 and R5 into R4_R5, then process exclusions.
    Expected branches:
        - Branch A (Excl(R1)): R1, R2, R3 removed; R4_R5 remains.
        - Branch B (Excl(R4_R5)): R4_R5 removed; R1, R2, R3 remain.
    """
    from src.domain.plugins.nrp import NRPPlugin
    from src.domain.core.pipeline import PreprocessingEngine

    model_with_transitivity_and_coupling.plugin = NRPPlugin
    steps = NRPPlugin.get_preprocessing_steps()

    final_branches = PreprocessingEngine.process(
        initial_model=model_with_transitivity_and_coupling,
        custom_steps=steps
    )

    assert len(final_branches) == 2

    # --- Branch A: Remove R1 ---
    branch_a = next(b for b in final_branches if "Excl(R1)" in b.branch_name)

    assert "R1" not in branch_a.items
    assert "R2" not in branch_a.items
    assert "R3" not in branch_a.items

    # R4 and R5 should be fused into R4_R5
    assert "R4" not in branch_a.items
    assert "R5" not in branch_a.items
    assert "R4_R5" in branch_a.items

    # --- Branch B: Remove R4 (now named Excl(R4_R5) because R4 and R5 are fused) ---
    branch_b = next(b for b in final_branches if "Excl(R4_R5)" in b.branch_name)

    # R4_R5 removed
    assert "R4_R5" not in branch_b.items
    assert "R4" not in branch_b.items
    assert "R5" not in branch_b.items

    # R1, R2, R3 remain
    assert "R1" in branch_b.items
    assert "R2" in branch_b.items
    assert "R3" in branch_b.items