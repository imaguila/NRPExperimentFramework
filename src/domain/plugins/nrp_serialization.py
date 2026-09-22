# Location: src/domain/plugins/nrp_serialization.py
"""
NRP Serialization utilities.

This module provides functions to serialise an OptimizationModel into
the NRP‑specific dictionary format and to export it to JSON. It also
includes helpers for normalising relationship pairs and custom JSON
serialisation of non‑primitive objects.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.core.model import OptimizationModel

from src.domain.plugins.nrp_schema import NRPAttributeSchema


def normalize_pairs(raw_list: List[Any]) -> List[Tuple[str, str]]:
    """
    Normalise a list of relationship pairs (dict, tuple, or list) into
    tuples of strings (src, tgt).

    Supported dictionary keys: 'from'/'to', 'req1'/'req2', 'r1'/'r2',
    'source'/'target'.

    Args:
        raw_list: List of items representing pairs.

    Returns:
        List of (source, target) string tuples.
    """
    normalized = []
    if not raw_list:
        return normalized
    for item in raw_list:
        if isinstance(item, dict):
            r1 = item.get("from") or item.get("req1") or item.get("r1") or item.get("source")
            r2 = item.get("to") or item.get("req2") or item.get("r2") or item.get("target")
            if r1 is not None and r2 is not None:
                normalized.append((str(r1), str(r2)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            normalized.append((str(item[0]), str(item[1])))
    return normalized


def json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for non‑primitive objects.

    Handles objects with a `to_dict()` method, dataclasses, sets, and falls
    back to string representation.

    Args:
        obj: The object to serialise.

    Returns:
        A JSON‑serialisable representation.
    """
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def export_to_dict(model: OptimizationModel) -> Dict[str, Any]:
    """
    Serialise an OptimizationModel to the strict NRP dictionary format.

    The output includes requirements, attribute definitions, relationships
    (precedences, couplings, exclusions, value dependencies), stakeholders,
    developers, and metadata. If the model has a plugin, it is not used here;
    this function produces a plain NRP‑compatible structure.

    Args:
        model: The OptimizationModel to serialise.

    Returns:
        A dictionary in the NRP canonical format.
    """
    # Local import to avoid circular dependency
    from src.domain.core.model import OptimizationModel  # noqa: F401

    schema = NRPAttributeSchema.get_attribute_schema()
    requirements = []
    items_dict = getattr(model, "items", {}) or {}

    for item_id, item in items_dict.items():
        if hasattr(item, "to_dict") and callable(item.to_dict):
            requirements.append(item.to_dict())
        elif isinstance(item, dict):
            req_dict = dict(item)
            if "id" not in req_dict:
                req_dict["id"] = str(item_id)
            requirements.append(req_dict)
        else:
            requirements.append({"id": str(item_id), "description": f"Requirement {item_id}", "attributes": {}})

    rels = getattr(model, "relationships", {}) or {}
    precedences = [{"req1": p[0], "req2": p[1]} for p in normalize_pairs(rels.get("precedences", []))]
    couplings = [{"req1": c[0], "req2": c[1]} for c in normalize_pairs(rels.get("couplings", []))]
    exclusions = [{"req1": e[0], "req2": e[1]} for e in normalize_pairs(rels.get("exclusions", []))]

    raw_data = getattr(model, "raw_data", {}) or {}
    evaluators = getattr(model, "evaluators", {}) or {}
    stakeholders = evaluators.get("stakeholders") or raw_data.get("stakeholders", [])
    developers = evaluators.get("developers") or raw_data.get("developers", [])

    val_deps = (
        getattr(model, "value_dependencies", None)
        or rels.get("value_dependencies")
        or raw_data.get("value_dependencies")
        or []
    )

    raw_attr_defs = getattr(model, "attribute_definitions", {}) or raw_data.get("attribute_definitions", {})
    attr_defs_dict = {}
    if isinstance(raw_attr_defs, dict) and raw_attr_defs:
        for k, v in raw_attr_defs.items():
            if hasattr(v, "to_dict") and callable(v.to_dict):
                v_dict = v.to_dict()
            elif hasattr(v, "__dict__"):
                v_dict = v.__dict__
            elif isinstance(v, dict):
                v_dict = v
            else:
                v_dict = {}
            schema_info = schema.get(k, {})
            attr_defs_dict[k] = {
                "type": v_dict.get("type", schema_info.get("type", "scalar")),
                "description": v_dict.get("description", schema_info.get("description", f"Attribute {k}")),
                "coupling_rule": v_dict.get("coupling_rule", schema_info.get("coupling_rule", "sum")),
                "aggregation_rule": v_dict.get("aggregation_rule", schema_info.get("aggregation_rule", "weighted_mean")),
                "evaluation_rule": v_dict.get("evaluation_rule", schema_info.get("evaluation_rule", "sum"))
            }
    else:
        used_attrs = set()
        for req in requirements:
            used_attrs.update(req.get("attributes", {}).keys())
        for attr in sorted(used_attrs):
            info = schema.get(attr, {})
            attr_defs_dict[attr] = {
                "type": info.get("type", "scalar"),
                "description": info.get("description", f"Attribute {attr}"),
                "coupling_rule": info.get("coupling_rule", "sum"),
                "aggregation_rule": info.get("aggregation_rule", "weighted_mean"),
                "evaluation_rule": info.get("evaluation_rule", "sum")
            }

    export_dict = {
        "name": getattr(model, "name", "NRP_Instance"),
        "num_requirements": len(requirements),
        "attribute_definitions": attr_defs_dict,
        "requirements": requirements,
        "relationships": {
            "precedences": precedences,
            "couplings": couplings,
            "exclusions": exclusions,
            "value_dependencies": val_deps
        },
        "value_dependencies": val_deps
    }
    if stakeholders:
        export_dict["stakeholders"] = stakeholders
    if developers:
        export_dict["developers"] = developers
    return export_dict


def export_to_json(model: OptimizationModel, indent: int = 2) -> str:
    """
    Serialise an OptimizationModel to a JSON string in the NRP format.

    Args:
        model: The model to serialise.
        indent: Indentation level for pretty‑printing.

    Returns:
        A JSON string.
    """
    data_dict = export_to_dict(model)
    return json.dumps(data_dict, indent=indent, ensure_ascii=False, default=json_serializer)