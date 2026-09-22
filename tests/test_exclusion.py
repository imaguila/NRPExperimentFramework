# Location: tests/test_exclusion.py
"""
Unit tests for the exclusion preprocessor.

These tests verify that the exclusion preprocessor correctly branches on
mutual exclusion constraints and cascades removal to items that *require*
the removed item (predecessors in the precedence graph, per the paper's
Eq. 9 Dep(i)) and to coupled items — never to items the removed item
itself requires (successors), which must be preserved since they may
still be needed independently.

NOTE: the shared `base_model` fixture in conftest.py has its precedence
comments phrased loosely ("R2 depende de R1" for {"req1": "R1", "req2":
"R2"}). Per the formal convention used throughout the framework (matching
Concise_Case_11.json's {"from": ..., "to": ...} and Eq. 1 of the paper:
selecting req1 requires req2), {"req1": "R1", "req2": "R2"} means "R1
requires R2", i.e. R1 is the one with the unsatisfiable condition if R2
is removed — not R2 depending on R1. The assertions below follow that
formal convention, which is also what `extract_pair`/`normalize_pairs`
implement (req1/from -> source, req2/to -> target).
"""

import pytest

from src.domain.core.item import DecisionItem
from src.domain.core.model import OptimizationModel
from src.domain.plugins import nrp_exclusion


def test_no_exclusions_return_same_model(sample_items):
    """
    Test that the preprocessor returns the original model unchanged
    when there are no exclusions.
    """
    model = OptimizationModel("NoExcl", sample_items, {"exclusions": []})
    branches = nrp_exclusion.run(model)

    assert len(branches) == 1
    assert branches[0].branch_name == "Base"


def test_exclusion_branching_and_cascade_deletion(base_model):
    """
    base_model precedences: R1 -> R2 -> R3, i.e. "R1 requires R2" and
    "R2 requires R3" (selecting R1 forces R2; selecting R2 forces R3).
    Coupling: R4-R5. Exclusion: R1 vs R4.

    Expected branches:
        - Excl(R1): only R1 is removed. Nothing has a precedence pointing
          *to* R1 (R1 is only ever a source), so nothing cascades on the
          precedence side. R2 and R3 are preserved: they are requirements
          *of* R1, not requirers *of* R1. Remaining: R2, R3, R4, R5.
        - Excl(R4): R4 removed, and R5 removed with it via coupling
          (unaffected by this fix — coupling is symmetric).
          Remaining: R1, R2, R3.
    """
    branches = nrp_exclusion.run(base_model)

    assert len(branches) == 2

    branch_excl_r1 = next(b for b in branches if "Excl(R1)" in b.branch_name)
    assert set(branch_excl_r1.items.keys()) == {"R2", "R3", "R4", "R5"}

    branch_excl_r4 = next(b for b in branches if "Excl(R4)" in b.branch_name)
    assert set(branch_excl_r4.items.keys()) == {"R1", "R2", "R3"}


@pytest.fixture
def model_with_dependent_predecessor():
    """
    A model where R2 requires R4 (precedence R2 -> R4), and R4 is excluded
    against R1. This is the case the *original* implementation got wrong:
    it only ever cascaded to successors (what the removed item requires),
    never to predecessors (what requires the removed item).
    """
    items = {
        "R1": DecisionItem("R1", "Requisito 1", {"cost": 10.0}),
        "R2": DecisionItem("R2", "Requisito 2", {"cost": 20.0}),
        "R4": DecisionItem("R4", "Requisito 4", {"cost": 30.0}),
    }
    relationships = {
        "precedences": [
            {"req1": "R2", "req2": "R4"},  # R2 requires R4
        ],
        "couplings": [],
        "exclusions": [
            {"req1": "R1", "req2": "R4"},
        ],
        "value_dependencies": [],
    }
    return OptimizationModel(
        name="Model_With_Dependent_Predecessor",
        items=items,
        relationships=relationships,
    )


def test_removing_a_required_item_cascades_to_its_requirer(model_with_dependent_predecessor):
    """
    Regression test for the direction of the cascade: R2 requires R4
    (R2 -> R4). When R4 is removed by the exclusion, R2's implication
    condition can never be satisfied again, so R2 must be removed too.

    The original (buggy) implementation missed this entirely — it only
    ever removed successors of the removed item, never predecessors — and
    would have kept R2 in this branch with a now-unenforceable precedence
    silently dropped from `relationships["precedences"]`.
    """
    branches = nrp_exclusion.run(model_with_dependent_predecessor)

    branch_excl_r4 = next(b for b in branches if "Excl(R4)" in b.branch_name)
    assert "R4" not in branch_excl_r4.items
    assert "R2" not in branch_excl_r4.items  # cascaded: R2 required R4
    assert set(branch_excl_r4.items.keys()) == {"R1"}

    branch_excl_r1 = next(b for b in branches if "Excl(R1)" in b.branch_name)
    # R1 removed; nothing requires R1, and R1 requires nothing here, so R2
    # and R4 (and the R2 -> R4 precedence) are untouched.
    assert set(branch_excl_r1.items.keys()) == {"R2", "R4"}
    assert branch_excl_r1.relationships["precedences"] == [
        {"req1": "R2", "req2": "R4"}
    ]


def test_no_dangling_precedences_after_branching(base_model):
    """Every surviving precedence must reference only surviving items —
    the fix must never leave a precedence pointing at a removed item."""
    branches = nrp_exclusion.run(base_model)
    for branch in branches:
        for rule in branch.relationships.get("precedences", []):
            assert rule["req1"] in branch.items
            assert rule["req2"] in branch.items
