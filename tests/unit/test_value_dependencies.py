# Location: tests/test_value_dependencies.py
"""
Unit tests for value dependencies / synergies (nrp_value_dependencies).

These tests verify that the value dependency preprocessor correctly handles
both intra‑item absorption (when coupled items are merged) and inter‑item
retention (when requirements remain separate).
"""

import pytest
from src.domain.plugins import nrp_coupling
from src.domain.plugins import nrp_value_dependencies


def test_intra_item_value_dependency_absorption(base_model):
    """
    Test that a value dependency between two coupled items is absorbed into
    the fused item's attributes.

    Steps:
        1. Couple R4 and R5 into R4_R5.
        2. Apply the value dependency that reduces the cost by 5 when both are included.
        3. The fused item's cost should be the original combined cost (55.0) minus 5.0 = 50.0.
        4. The dependency should be removed from the remaining dependencies list
           because it was statically absorbed.
    """
    # 1. First couple R4 and R5 into R4_R5
    coupled_model = nrp_coupling.run(base_model)[0]

    # 2. The value dependency was for ["R4", "R5"] with a delta of -5 on 'cost'
    results = nrp_value_dependencies.run(coupled_model)
    processed_model = results[0]

    # The base combined cost was 55.0. After absorbing the delta of -5.0, it should be 50.0.
    fused_item = processed_model.items["R4_R5"]
    assert fused_item.attributes["cost"] == 50.0

    # Since it was statically absorbed, it should not remain in value_dependencies
    remaining_deps = processed_model.relationships["value_dependencies"]
    # Only the inter‑item synergy (R1, R3) should remain
    assert len(remaining_deps) == 1


def test_inter_item_value_dependency_retention(base_model):
    """
    Test that a value dependency between items that are not coupled is retained
    for dynamic evaluation.

    In the base model, R1 and R3 have a synergy multiplier of 1.2 on 'satisfaction'.
    Since they are not coupled, the dependency should remain in the relationships.
    """
    results = nrp_value_dependencies.run(base_model)
    processed_model = results[0]

    # R1 and R3 are not fused, so the rule should remain for dynamic evaluation
    deps = processed_model.relationships["value_dependencies"]
    assert any(d.get("reqs") == ["R1", "R3"] for d in deps)


def test_multiplier_semantics_preserved_after_coupling():
    """
    A multiplier applied dynamically before coupling and
    statically absorbed after coupling must produce the
    same objective value.
    """

    from src.domain.core.item import DecisionItem
    from src.domain.core.model import OptimizationModel
    from src.solving.core.evaluator import SolutionEvaluator

    # Original model
    items = {
        "r1": DecisionItem(
            id="r1",
            description="r1",
            attributes={"value": 10}
        ),
        "r2": DecisionItem(
            id="r2",
            description="r2",
            attributes={"value": 20}
        ),
        "r3": DecisionItem(
            id="r3",
            description="r3",
            attributes={"value": 100}
        )
    }

    relationships = {
        "precedences": [],
        "couplings": [],
        "exclusions": [],
        "value_dependencies": [
            {
                "attribute": "value",
                "reqs": ["r1", "r2"],
                "effect": "multiplier",
                "factor": 2.0,
            }
        ],
    }

    model_before = OptimizationModel(
        name="before",
        items=items,
        relationships=relationships,
    )

    evaluator_before = SolutionEvaluator(model_before)

    value_before = evaluator_before.evaluate_attribute(
        {"r1", "r2", "r3"},
        "value"
    )

    # Coupling absorbed version
    items_after = {
        "r1_r2": DecisionItem(
            id="r1_r2",
            description="merged",
            attributes={"value": 60}  # (10+20)*2
        ),
        "r3": DecisionItem(
            id="r3",
            description="r3",
            attributes={"value": 100}
        )
    }

    relationships_after = {
        "precedences": [],
        "couplings": [],
        "exclusions": [],
        "value_dependencies": []
    }

    model_after = OptimizationModel(
        name="after",
        items=items_after,
        relationships=relationships_after,
    )

    evaluator_after = SolutionEvaluator(model_after)

    value_after = evaluator_after.evaluate_attribute(
        {"r1_r2", "r3"},
        "value"
    )

    assert value_before == value_after
    assert value_before == 160.0

def test_effort_multiplier_affects_only_activating_subset():
    """
    Dynamic multipliers must affect only the activating subset,
    not the complete release value.
    """

    from src.domain.core.item import DecisionItem
    from src.domain.core.model import OptimizationModel
    from src.solving.core.evaluator import SolutionEvaluator

    items = {
        "r6": DecisionItem(
            id="r6",
            description="r6",
            attributes={"effort": 3}
        ),
        "r11": DecisionItem(
            id="r11",
            description="r11",
            attributes={"effort": 5}
        ),
        "r9": DecisionItem(
            id="r9",
            description="r9",
            attributes={"effort": 3}
        ),
    }

    relationships = {
        "precedences": [],
        "couplings": [],
        "exclusions": [],
        "value_dependencies": [
            {
                "attribute": "effort",
                "reqs": ["r6", "r11"],
                "effect": "multiplier",
                "factor": 0.9,
            }
        ],
    }

    model = OptimizationModel(
        name="multiplier_test",
        items=items,
        relationships=relationships,
    )

    evaluator = SolutionEvaluator(model)

    computed_value = evaluator.evaluate_attribute(
        {"r6", "r11", "r9"},
        "effort"
    )

    expected = ((3 + 5) * 0.9) + 3

    assert abs(computed_value - expected) < 1e-9

    # Ensure the multiplier is NOT applied to the whole release:
    assert computed_value != (3 + 5 + 3) * 0.9