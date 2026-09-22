# Location: tests/test_coupling.py
"""
Unit tests for the coupling preprocessor.

These tests verify that the coupling preprocessor correctly merges coupled
items, applies custom aggregation rules, and remaps relationships accordingly.
"""

import pytest
from src.domain.plugins import nrp_coupling


def test_coupling_fusion(base_model):
    """
    Test that coupled items are correctly fused into a single unified item.

    In the base_model, R4 and R5 are coupled. The preprocessor should replace
    them with a single item R4_R5, with attributes aggregated using default rules.
    """
    # Run the coupling preprocessor on the base model
    results = nrp_coupling.run(base_model)
    assert len(results) == 1

    model_coupled = results[0]

    # Verify that R4 and R5 are replaced by R4_R5
    assert "R4" not in model_coupled.items
    assert "R5" not in model_coupled.items
    assert "R4_R5" in model_coupled.items

    # Verify attribute aggregation (default rule: sum)
    fused_item = model_coupled.items["R4_R5"]
    assert fused_item.attributes["cost"] == 55.0   # 30 + 25
    assert fused_item.attributes["time"] == 16.0   # 12 + 4


def test_coupling_rule_custom_aggregation(base_model):
    """
    Test that custom aggregation rules are applied correctly.

    For the 'time' attribute, the max rule is used instead of the default sum.
    """
    custom_rules = {"time": "max", "cost": "sum"}
    results = nrp_coupling.run(base_model, custom_couplings=custom_rules)
    model_coupled = results[0]

    fused_item = model_coupled.items["R4_R5"]
    # time should be max(12, 4) = 12, not 16
    assert fused_item.attributes["time"] == 12.0


def test_coupling_remapped_relationships(base_model):
    """
    Test that relationships are correctly remapped after coupling.

    The exclusion between R1 and R4 should be updated to reference R1 and R4_R5,
    since R4 has been fused with R5.
    """
    results = nrp_coupling.run(base_model)
    model_coupled = results[0]

    # The exclusion should now reference R4_R5 instead of R4
    exclusions = model_coupled.relationships["exclusions"]
    assert len(exclusions) == 1
    assert exclusions[0] == {"req1": "R1", "req2": "R4_R5"}