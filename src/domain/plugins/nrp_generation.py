# Location: src/domain/plugins/nrp_generation.py

"""
NRP‑specific generation logic.

This module provides functions to generate synthetic instances of the
Next Release Problem (NRP) domain. It handles the creation of items,
attribute sampling, precedence injection, and validation of the resulting
Directed Acyclic Graph (DAG).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from src.domain.core.model import OptimizationModel
from src.domain.core.definitions import AttributeDefinition
from src.domain.plugins.nrp import NRPPlugin
from src.domain.plugins.nrp_graph import build_precedence_graph, validate_dag
from src.domain.core.generators.precedence_generator import inject_precedences_into_model


def _json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for non‑primitive objects.

    Handles objects with a `to_dict()` method, dataclasses, sets, and falls
    back to string representation.
    """
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def generate_nrp_instance(
    num_items: int,
    attribute_configs: Dict[str, Dict[str, Any]],
    value_dependencies: List[Dict[str, Any]],
    enable_precedences: bool = True,
    precedence_density: float = 0.10,
) -> OptimizationModel:
    """
    Generate a synthetic NRP model instance.

    The function creates a specified number of items (requirements) with
    attributes sampled according to the provided configurations. It also
    injects value dependencies and optionally generates precedence relationships
    as a DAG.

    Args:
        num_items: Number of requirements to generate.
        attribute_configs: Mapping of attribute names to their generation
            configurations (distribution, min, max, etc.).
        value_dependencies: List of value dependency specifications.
        enable_precedences: Whether to generate precedence relationships.
        precedence_density: Density of precedence edges (0.0 to 1.0).

    Returns:
        An OptimizationModel instance representing the generated NRP instance.

    Raises:
        ValueError: If the generated precedence graph contains cycles.
    """
    model = OptimizationModel(
        name=f"Synthetic_NRP_{num_items}_Items",
        items={},
        attribute_definitions={},
        relationships={"precedences": [], "couplings": [], "exclusions": [], "value_dependencies": []},
        raw_data={},
        plugin=NRPPlugin,
    )

    # Generate items with sampled attributes
    item_id_prefix = "R"
    for i in range(1, num_items + 1):
        item_id = f"{item_id_prefix}{i}"
        attrs = {}
        for attr_name, attr_config in attribute_configs.items():
            attrs[attr_name] = _sample_attribute_value(attr_config)

        from src.domain.core.item import DecisionItem
        model.items[item_id] = DecisionItem(
            id=item_id,
            description=f"Requirement {i}",
            attributes=attrs,
        )

    # Build attribute definitions from the schema
    schema = NRPPlugin.get_attribute_schema()
    selected_defs = {}
    for attr_id in attribute_configs.keys():
        if attr_id in schema:
            info = schema[attr_id]
            selected_defs[attr_id] = AttributeDefinition(
                name=attr_id,
                type=info.get("type", "scalar"),
                description=info.get("description", f"Attribute {attr_id}"),
                evaluator_group=info.get("evaluator_group"),
                coupling_rule=info.get("coupling_rule", "sum"),
                aggregation_rule=info.get("aggregation_rule", "weighted_mean"),
                evaluation_rule=info.get("evaluation_rule", "sum"),
            )

    model.attribute_definitions = selected_defs
    model.relationships["value_dependencies"] = value_dependencies
    model.raw_data["attribute_definitions"] = selected_defs
    model.raw_data["value_dependencies"] = value_dependencies

    # Optionally inject precedence relationships
    if enable_precedences and precedence_density > 0:
        model = inject_precedences_into_model(model, density=precedence_density)
        model.plugin = NRPPlugin

        graph = build_precedence_graph(model)
        if not validate_dag(graph):
            raise ValueError("The generated precedence graph contains cycles.")

    return model


def _sample_attribute_value(config: Dict[str, Any]) -> Any:
    """
    Sample a value for an attribute according to the given configuration.

    Supported distributions: constant, uniform, normal, triangular.
    The value can be forced to integer if `is_integer` is True.

    Args:
        config: Dictionary containing distribution type and parameters.

    Returns:
        A sampled value (int or float).
    """
    import random

    dist_type = str(config.get("distribution", "uniform")).lower()
    is_integer = bool(config.get("is_integer", True))

    if dist_type == "constant":
        value = config.get("value", 10)
        return int(value) if is_integer else float(value)

    if dist_type == "uniform":
        low = config.get("min", 1)
        high = config.get("max", 100)
        return random.randint(int(low), int(high)) if is_integer else random.uniform(low, high)

    if dist_type == "normal":
        mean = config.get("mean", 50.0)
        std = config.get("std", 15.0)
        min_bound = config.get("min", 1.0)
        max_bound = config.get("max", 200.0)
        value = max(min_bound, min(max_bound, random.gauss(mean, std)))
        return round(value) if is_integer else round(value, 2)

    if dist_type == "triangular":
        low = config.get("min", 1.0)
        mode = config.get("mode", 50.0)
        high = config.get("max", 100.0)
        value = random.triangular(low, high, mode)
        return round(value) if is_integer else round(value, 2)

    # Fallback to uniform integer
    low = config.get("min", 1)
    high = config.get("max", 100)
    return random.randint(int(low), int(high)) if is_integer else random.uniform(low, high)


def get_nrp_generation_config() -> Dict[str, Any]:
    """
    Return the default configuration parameters for NRP instance generation.

    Returns:
        Dictionary with parameter names, min/max/default values, and labels.
    """
    return {
        "num_requirements": {
            "min": 5,
            "max": 1000,
            "default": 20,
            "step": 5,
            "label": "Number of Requirements",
        },
        "precedence_density": {
            "min": 0.0,
            "max": 0.5,
            "default": 0.05,
            "step": 0.01,
            "enabled_by_default": True,
            "label": "Precedence density",
        },
    }


def model_to_nrp_dict(model: OptimizationModel) -> Dict[str, Any]:
    """
    Convert an OptimizationModel to a simplified NRP‑specific dictionary.

    This function extracts attribute definitions, item attributes, and
    precedence relationships in a format suitable for JSON export.

    Args:
        model: The OptimizationModel to convert.

    Returns:
        A dictionary with 'name', 'attribute_definitions', 'items', and
        'precedences' keys.
    """
    attr_defs = getattr(model, "attribute_definitions", {})
    attr_defs_dict = {}
    for k, v in attr_defs.items():
        if hasattr(v, "to_dict") and callable(v.to_dict):
            attr_defs_dict[k] = v.to_dict()
        elif isinstance(v, dict):
            attr_defs_dict[k] = v
        else:
            attr_defs_dict[k] = str(v)

    items_dict = {}
    for item_id, item_obj in model.items.items():
        attrs = getattr(item_obj, "attributes", {})
        clean_item = {}
        for attr_k, attr_v in attrs.items():
            if isinstance(attr_v, (int, float)):
                clean_item[attr_k] = attr_v
        items_dict[item_id] = clean_item

    precedences = []
    rel_precedences = model.relationships.get("precedences", [])
    for rel in rel_precedences:
        if isinstance(rel, dict) and "source" in rel and "target" in rel:
            precedences.append({"source": rel["source"], "target": rel["target"]})

    return {
        "name": getattr(model, "name", "NRP_Instance"),
        "attribute_definitions": attr_defs_dict,
        "items": items_dict,
        "precedences": precedences,
    }


def export_nrp_instance_to_json(model: OptimizationModel, filepath: Union[str, Path]) -> None:
    """
    Export an NRP model to a JSON file.

    The model is converted to a simplified dictionary using `model_to_nrp_dict`
    and then written to the specified path.

    Args:
        model: The OptimizationModel to export.
        filepath: Destination file path (parent directories are created if needed).
    """
    data = model_to_nrp_dict(model)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_serializer)