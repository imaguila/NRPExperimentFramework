# Location: src/domain/plugins/nrp_value_dependencies.py
"""
Value Dependency Preprocessor for NRP.

This module handles non‑linear value interactions (synergies) between requirements.
It resolves value dependencies by either absorbing them into a single item
(when all involved requirements are merged) or keeping them for dynamic
evaluation when requirements remain separate.
"""

from __future__ import annotations

import copy
from typing import List, Dict, Any

from src.domain.core.item import DecisionItem


def run(model, **kwargs) -> List:
    """
    Apply the value dependency preprocessor to the given model.

    This function processes value dependencies (synergies) defined in the model's
    relationships. For each dependency, it resolves the affected requirement IDs
    to their current item IDs (accounting for previous merges). If all involved
    requirements map to a single unified item, the dependency is absorbed
    statically by modifying the item's attribute. Otherwise, the dependency is
    kept for dynamic evaluation with the resolved item IDs.

    Args:
        model: The OptimizationModel to process.
        **kwargs: Additional arguments (unused).

    Returns:
        A list containing a single new OptimizationModel with value dependencies
        either absorbed or remapped.
    """
    from src.domain.core.model import OptimizationModel  # Local import to avoid circular dependency

    val_deps = model.relationships.get("value_dependencies", [])
    if not val_deps:
        return [model]

    new_items = copy.deepcopy(model.items)
    remaining_val_deps = []

    for dep in val_deps:
        if not isinstance(dep, dict):
            continue
        reqs = dep.get("reqs", [])
        attr = str(dep.get("attribute", "")).strip()
        effect = str(dep.get("effect", "delta")).strip().lower()
        factor = float(dep.get("factor", 0.0))

        if not reqs or not attr:
            remaining_val_deps.append(dep)
            continue

        # Resolve requirement IDs to current item IDs (account for coupling merges)
        resolved_items = []
        for r in reqs:
            r_str = str(r).strip()
            if r_str in new_items:
                resolved_items.append(r_str)
            else:
                # Try to find an item that contains this ID (e.g., "R4_R5" contains "R4")
                found_id = next((item_id for item_id in new_items.keys() if r_str in item_id.split("_")), r_str)
                resolved_items.append(found_id)

        unique_targets = set(resolved_items)

        # Case 1: All requirements resolved to a single item → absorb statically
        if len(unique_targets) == 1:
            target_id = list(unique_targets)[0]
            if target_id in new_items:
                target_item = new_items[target_id]
                item_attrs = getattr(target_item, "attributes", {})
                if attr in item_attrs:
                    curr_val = item_attrs[attr]
                    if effect in ("multiplier", "multiply", "scale"):
                        new_val = float(curr_val * factor)
                    else:  # delta
                        new_val = float(curr_val + factor)
                    updated_attrs = dict(item_attrs)
                    updated_attrs[attr] = new_val
                    new_items[target_id] = DecisionItem(
                        id=target_item.id,
                        description=getattr(target_item, "description", target_id),
                        attributes=updated_attrs
                    )
                else:
                    # Attribute not present → create it with the factor value
                    updated_attrs = dict(item_attrs)
                    updated_attrs[attr] = factor
                    new_items[target_id] = DecisionItem(
                        id=target_item.id,
                        description=getattr(target_item, "description", target_id),
                        attributes=updated_attrs
                    )
        # Case 2: Multiple target items → keep dependency for dynamic evaluation
        else:
            dep_copy = copy.deepcopy(dep)
            dep_copy["reqs"] = list(dict.fromkeys(resolved_items))  # preserve order, remove duplicates
            remaining_val_deps.append(dep_copy)

    # Update relationships with the remaining dynamic dependencies
    new_relationships = copy.deepcopy(model.relationships)
    new_relationships["value_dependencies"] = remaining_val_deps

    # Build and return the new model
    return [
        OptimizationModel(
            name=model.name,
            items=new_items,
            attribute_definitions=copy.deepcopy(
                getattr(model, "attribute_definitions", {})
            ),
            evaluators=copy.deepcopy(
                getattr(model, "evaluators", {})
            ),
            relationships=new_relationships,
            metadata=copy.deepcopy(
                getattr(model, "metadata", {})
            ),
            raw_data=copy.deepcopy(
                getattr(model, "raw_data", {})
            ),
            plugin=getattr(model, "plugin", None),
        )
    ]
