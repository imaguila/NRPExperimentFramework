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