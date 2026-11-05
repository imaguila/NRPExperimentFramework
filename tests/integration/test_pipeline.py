# Location: tests/test_pipeline.py
"""
Integration tests for the full preprocessing pipeline.

These tests verify that the preprocessing engine correctly executes
the configured pipeline steps (coupling, value dependencies, exclusion)
and produces the expected model variants and attribute aggregations.
"""

import pytest
from src.domain.core.pipeline import PreprocessingEngine
from src.domain.plugins.nrp import NRPPlugin
from src.domain.plugins import nrp_coupling
from src.domain.plugins import nrp_value_dependencies


def test_full_pipeline_execution(base_model):
    """
    Test that the full preprocessing pipeline executes correctly.

    The pipeline should run coupling, value dependencies, and exclusion steps.
    The final branches should have no couplings and the fused item R4_R5
    should have the correct aggregated cost (50.0).
    """
    base_model.plugin = NRPPlugin
    steps = NRPPlugin.get_preprocessing_steps()
    final_branches = PreprocessingEngine.process(
        initial_model=base_model,
        custom_steps=steps
    )
    assert len(final_branches) == 2
    for branch in final_branches:
        assert branch.relationships.get("couplings") == []
        if "R4_R5" in branch.items:
            assert branch.items["R4_R5"].attributes["cost"] == 50.0


def test_intra_item_value_dependency_absorption(base_model):
    """
    Test that value dependencies between coupled items are absorbed
    into the fused item's attributes.

    Steps:
        1. Couple R4 and R5 into a single item R4_R5.
        2. Apply the value dependency that reduces the cost by 5 when both are included.
        3. The fused item's cost should be the original combined cost (55.0) minus 5.0 = 50.0.
    """
    # 1. First couple R4 and R5 into R4_R5
    coupled_model = nrp_coupling.run(base_model)[0]

    # 2. The value dependency was for ["R4", "R5"] with a delta of -5 on 'cost'
    results = nrp_value_dependencies.run(coupled_model)
    processed_model = results[0]

    # The base combined cost was 55.0. After absorbing the delta of -5.0, it should be 50.0.
    fused_item = processed_model.items["R4_R5"]
    assert fused_item.attributes["cost"] == 50.0