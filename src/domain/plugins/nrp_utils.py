# src/domain/plugin/nrp_utils.py
"""
Relationship extraction utilities.

This module provides a function to extract source-target pairs from various
input formats (dict, tuple, list) commonly used to represent relationships
such as precedences, couplings, and exclusions.
"""

from typing import Any, Tuple


def extract_pair(item: Any) -> Tuple[str, str]:
    """
    Extract a pair of strings (source, target) from a relationship item.

    Supports:
        - tuple/list of length >= 2
        - dict with keys: 'from'/'to', 'req1'/'req2', 'r1'/'r2', 'source'/'target'

    Args:
        item: A tuple, list, or dict representing a relationship.

    Returns:
        A tuple of two strings (source, target). Returns ("", "") if extraction fails.
    """
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return str(item[0]).strip(), str(item[1]).strip()

    if isinstance(item, dict):
        src = (
            item.get("req1")
            or item.get("from")
            or item.get("r1")
            or item.get("source")
            or ""
        )
        tgt = (
            item.get("req2")
            or item.get("to")
            or item.get("r2")
            or item.get("target")
            or ""
        )
        return str(src).strip(), str(tgt).strip()

    return "", ""