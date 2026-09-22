"""
Tests for the headless JSON case-loading application use case.
"""

import json

from src.application.use_cases.load_json_case import (
    inspect_json_case,
    load_json_case,
)
from src.domain.plugins.nrp import NRPPlugin


def _build_test_instance() -> dict:
    """
    Build a small instance containing multivalued attributes and coupling.
    """
    return {
        "name": "Application layer test",
        "attribute_definitions": {
            "effort": {
                "type": "scalar",
                "coupling_rule": "sum",
                "evaluation_rule": "sum",
            },
            "satisfaction": {
                "type": "multivalued",
                "evaluator_group": "stakeholders",
                "aggregation_rule": "weighted_mean",
                "coupling_rule": "sum",
                "evaluation_rule": "sum",
            },
        },
        "stakeholders": [
            {
                "id": "S1",
                "weight": 1.0,
            },
            {
                "id": "S2",
                "weight": 3.0,
            },
        ],
        "requirements": [
            {
                "id": "R1",
                "attributes": {
                    "effort": 10.0,
                    "satisfaction": [2.0, 6.0],
                },
            },
            {
                "id": "R2",
                "attributes": {
                    "effort": 20.0,
                    "satisfaction": [4.0, 8.0],
                },
            },
        ],
        "relationships": {
            "precedences": [],
            "couplings": [
                {
                    "req1": "R1",
                    "req2": "R2",
                }
            ],
            "exclusions": [],
            "value_dependencies": [],
        },
    }


def test_inspect_json_case_returns_model():
    json_input = json.dumps(
        _build_test_instance()
    )

    inspection = inspect_json_case(
        json_input=json_input,
        plugin=NRPPlugin,
    )

    assert inspection.raw_data["name"] == (
        "Application layer test"
    )

    assert len(
        inspection.inspected_model.items
    ) == 2


def test_load_json_case_executes_preprocessing():
    json_input = json.dumps(
        _build_test_instance()
    )

    result = load_json_case(
        json_input=json_input,
        plugin=NRPPlugin,
    )

    assert result.raw_data["name"] == (
        "Application layer test"
    )

    assert len(result.base_model.items) == 2
    assert len(result.processed_models) == 1

    processed_model = result.processed_models[0]

    assert set(processed_model.items) == {
        "R1_R2"
    }

    assert (
        processed_model.relationships[
            "couplings"
        ]
        == []
    )


def test_load_json_case_applies_aggregation_overrides():
    json_input = json.dumps(
        _build_test_instance()
    )

    result = load_json_case(
        json_input=json_input,
        plugin=NRPPlugin,
        aggregation_overrides={
            "satisfaction": "mean",
        },
    )

    base_model = result.base_model

    assert (
        base_model.items[
            "R1"
        ].attributes["satisfaction"]
        == 4.0
    )


def test_application_use_case_is_headless():
    import src.application.use_cases.load_json_case as module

    module_source_names = set(
        module.__dict__.keys()
    )

    assert "streamlit" not in module_source_names
    assert "st" not in module_source_names