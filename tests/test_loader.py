# Location: tests/test_loader.py
"""
Tests for the GenericJSONLoader.

These tests verify that the loader correctly deserialises JSON data into
OptimizationModel instances, handles weighted aggregation of multivalued
attributes, and supports custom aggregation rule overrides.
"""

import json
import pytest
from src.domain.core.loader import GenericJSONLoader
from src.domain.plugins.nrp import NRPPlugin


def test_loader_creates_model_from_json(tmp_path):
    """
    Test that GenericJSONLoader correctly loads a realistic JSON case.

    The test verifies:
        - Model name and item count.
        - Stakeholder and developer evaluators are loaded with correct weights.
        - Relationships (precedences and couplings) are preserved.
        - Weighted mean aggregation is applied to multivalued 'satisfaction'
          attributes using stakeholder weights.

    Expected weighted means:
        - r1: satisfaction [3,4] with weights [2.0, 1.0] → (3*2 + 4*1)/3 ≈ 3.333
        - r2: satisfaction [1,2] with weights [2.0, 1.0] → (1*2 + 2*1)/3 ≈ 1.333
    """
    # Build a realistic JSON case
    json_data = {
        "name": "Test Case",
        "requirements": [
            {
                "id": "r1",
                "description": "Req 1",
                "attributes": {"effort": 5, "satisfaction": [3, 4]}
            },
            {
                "id": "r2",
                "description": "Req 2",
                "attributes": {"effort": 3, "satisfaction": [1, 2]}
            }
        ],
        # Two stakeholders for weighted aggregation
        "stakeholders": [
            {"id": "c1", "weight": 2.0, "description": "Primary client"},
            {"id": "c2", "weight": 1.0, "description": "Secondary client"}
        ],
        # Two developers
        "developers": [
            {"id": "d1", "weight": 1.0, "description": "Senior dev"},
            {"id": "d2", "weight": 0.5, "description": "Junior dev"}
        ],
        "relationships": {
            "precedences": [{"from": "r1", "to": "r2"}],
            "couplings": [{"req1": "r1", "req2": "r2"}]
        },
        # Attribute definitions to trigger weighted aggregation
        "attribute_definitions": {
            "effort": {"type": "scalar", "coupling_rule": "sum"},
            "satisfaction": {
                "type": "multivalued",
                "evaluator_group": "stakeholders",
                "aggregation_rule": "weighted_mean",
                "coupling_rule": "sum"
            }
        }
    }

    # Save the JSON to a temporary file
    json_path = tmp_path / "test_case.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f)

    # Load the model using GenericJSONLoader
    with open(json_path, "r") as f:
        json_str = f.read()

    plugin_instance = NRPPlugin()
    model = GenericJSONLoader.load(json_str, plugin=plugin_instance)

    # --- Verify basic model properties ---
    assert model.name == "Test Case"
    assert len(model.items) == 2
    assert "r1" in model.items
    assert "r2" in model.items

    # Verify that evaluators (stakeholders) are loaded correctly
    assert len(model.evaluators.get("stakeholders", [])) == 2
    assert model.evaluators["stakeholders"][0].id == "c1"
    assert model.evaluators["stakeholders"][0].weight == 2.0
    assert model.evaluators["stakeholders"][1].id == "c2"
    assert model.evaluators["stakeholders"][1].weight == 1.0

    # Verify relationships
    assert len(model.relationships.get("precedences", [])) == 1
    assert len(model.relationships.get("couplings", [])) == 1

    # --- Critical: verify weighted mean aggregation ---
    # r1: satisfaction = [3,4], weights [2.0, 1.0] → (3*2 + 4*1) / 3 ≈ 3.333
    r1_attrs = model.items["r1"].attributes
    assert "satisfaction" in r1_attrs
    assert abs(r1_attrs["satisfaction"] - 3.3333333333333335) < 1e-6

    # r2: satisfaction = [1,2], weights [2.0, 1.0] → (1*2 + 2*1) / 3 ≈ 1.333
    r2_attrs = model.items["r2"].attributes
    assert "satisfaction" in r2_attrs
    assert abs(r2_attrs["satisfaction"] - 1.3333333333333333) < 1e-6


def test_loader_custom_multivalued_aggregation_override():
    """
    Test that the loader accepts custom aggregation overrides.

    The default aggregation rule for 'satisfaction' is 'mean' in the JSON.
    By passing attr_aggregations={"satisfaction": "max"}, the loader should
    override the rule and use max instead.
    """
    json_data = {
        "name": "Custom Aggregation Case",
        "requirements": [
            {"id": "r1", "attributes": {"satisfaction": [1, 9, 7]}},
            {"id": "r2", "attributes": {"satisfaction": [4, 4]}},
        ],
        "attribute_definitions": {
            "satisfaction": {
                "type": "multivalued",
                "aggregation_rule": "mean",
                "coupling_rule": "sum",
                "evaluation_rule": "sum",
            }
        },
    }

    # Load with default aggregation (mean)
    model_default = GenericJSONLoader.load(json.dumps(json_data), plugin=NRPPlugin())

    # Load with custom aggregation override (max)
    model_custom = GenericJSONLoader.load(
        json.dumps(json_data),
        plugin=NRPPlugin(),
        attr_aggregations={"satisfaction": "max"},
    )

    # Default: mean of [1,9,7] = 17/3 ≈ 5.6667
    assert abs(model_default.items["r1"].attributes["satisfaction"] - 5.666666666666667) < 1e-6

    # Custom: max of [1,9,7] = 9
    assert model_custom.items["r1"].attributes["satisfaction"] == 9.0

    # For r2: [4,4] both mean and max give 4.0
    assert model_custom.items["r2"].attributes["satisfaction"] == 4.0