# Location: src/domain/plugins/nrp_extraction.py
"""
NRP Data Extraction utilities.

This module provides functions to extract domain‑specific data from raw JSON
dictionaries for the Next Release Problem (NRP) domain. It handles the
conversion of raw requirements, relationships, and evaluators into their
corresponding domain objects (DecisionItem, Evaluator, etc.).
"""

from typing import Any, Dict, List, Tuple

from src.domain.core.definitions import Evaluator
from src.domain.core.item import DecisionItem
from src.domain.plugins.nrp_serialization import normalize_pairs


def extract_items(data: Dict[str, Any]) -> Dict[str, DecisionItem]:
    """
    Extract DecisionItem objects from the raw JSON data.

    The function looks for a 'requirements' or 'items' key in the data
    dictionary. Each raw item is expected to have an 'id' field, and may
    optionally have 'description' and 'attributes' fields.

    Args:
        data: Raw dictionary loaded from JSON.

    Returns:
        Dictionary mapping item IDs to DecisionItem instances.
    """
    raw_reqs = data.get("requirements", []) or data.get("items", [])
    return {
        str(r["id"]): DecisionItem(
            id=str(r["id"]),
            description=r.get("description", ""),
            attributes=r.get("attributes", {}),
        )
        for r in raw_reqs if isinstance(r, dict) and "id" in r
    }


def extract_relationships(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relationships and constraints from the raw JSON data.

    The function retrieves precedences, couplings, exclusions, and value
    dependencies from the 'relationships' key (or directly from the root
    data for value dependencies). Pairs are normalised using `normalize_pairs`.

    Args:
        data: Raw dictionary loaded from JSON.

    Returns:
        Dictionary containing lists of normalised relationship pairs and
        value dependencies.
    """
    rels = data.get("relationships", {})
    val_deps = rels.get("value_dependencies", []) or data.get("value_dependencies", [])
    return {
        "precedences": normalize_pairs(rels.get("precedences", []) or rels.get("precedence", [])),
        "couplings": normalize_pairs(rels.get("couplings", []) or rels.get("coupling", [])),
        "exclusions": normalize_pairs(rels.get("exclusions", []) or rels.get("exclusion", [])),
        "value_dependencies": val_deps,
    }


def extract_evaluators(data: Dict[str, Any]) -> Dict[str, List[Evaluator]]:
    """
    Extract evaluators (stakeholders and developers) from the raw JSON data.

    The function looks for 'stakeholders' and 'developers' keys in the data.
    Each evaluator is converted into an Evaluator object with its ID, weight,
    description, and group set appropriately.

    Args:
        data: Raw dictionary loaded from JSON.

    Returns:
        Dictionary with 'stakeholders' and 'developers' keys, each mapping
        to a list of Evaluator objects.
    """
    stakeholders = data.get("stakeholders", [])
    developers = data.get("developers", [])
    return {
        "stakeholders": [
            Evaluator(
                id=str(e.get("id", "")),
                weight=float(e.get("weight", 1.0)),
                description=str(e.get("description", "")),
                group="stakeholders"
            ) for e in stakeholders
        ],
        "developers": [
            Evaluator(
                id=str(e.get("id", "")),
                weight=float(e.get("weight", 1.0)),
                description=str(e.get("description", "")),
                group="developers"
            ) for e in developers
        ]
    }