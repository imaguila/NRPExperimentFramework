# Location: tests/test_nrp_export.py
"""
Tests for NRP model export functionality.

These tests verify that synthetic NRP instances can be generated, exported
to dictionaries, and written to JSON files correctly.
"""

import json
import pytest
from pathlib import Path

from src.domain.plugins.nrp import NRPPlugin
from src.domain.plugins.nrp_generation import (
    generate_nrp_instance,
    model_to_nrp_dict,
    export_nrp_instance_to_json
)


@pytest.fixture
def sample_nrp_model():
    """Generate a synthetic NRP model instance for testing."""
    attr_configs = {
        "cost": {"distribution": "uniform", "min": 10.0, "max": 50.0},
        "value": {"distribution": "uniform", "min": 20.0, "max": 100.0}
    }
    value_dependencies = []

    return generate_nrp_instance(
        num_items=5,
        attribute_configs=attr_configs,
        value_dependencies=value_dependencies,
        enable_precedences=True,
        precedence_density=0.20
    )


def test_nrp_model_to_dict_structure(sample_nrp_model):
    """
    Verify that the generated dictionary contains only scalar attributes,
    items, and precedences.

    The dictionary should have:
        - 'name', 'attribute_definitions', 'items', and 'precedences' keys.
        - Exactly 5 items.
        - All item attributes are numeric scalars (int or float).
        - Precedences have 'source' and 'target' keys.
    """
    nrp_dict = model_to_nrp_dict(sample_nrp_model)

    assert "name" in nrp_dict
    assert "attribute_definitions" in nrp_dict
    assert "items" in nrp_dict
    assert "precedences" in nrp_dict

    # Verify that there are exactly 5 items
    assert len(nrp_dict["items"]) == 5

    # Verify that all item attributes are numeric scalars
    for item_id, attributes in nrp_dict["items"].items():
        for attr_k, attr_v in attributes.items():
            assert isinstance(attr_v, (int, float))

    # Verify that precedences have 'source' and 'target' keys
    for prec in nrp_dict["precedences"]:
        assert "source" in prec
        assert "target" in prec


def test_export_nrp_instance_to_json_file(sample_nrp_model, tmp_path):
    """
    Verify that exporting to JSON creates a valid, parseable file.

    The file should exist, contain the correct model name,
    and have the expected number of items and a list of precedences.
    """
    file_path = tmp_path / "exported_instance.json"

    export_nrp_instance_to_json(sample_nrp_model, file_path)
    assert file_path.exists()

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["name"] == sample_nrp_model.name
    assert len(data["items"]) == 5
    assert isinstance(data["precedences"], list)