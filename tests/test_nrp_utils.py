# Location: tests/test_nrp_utils.py
"""
Unit tests for common utilities (nrp_utils).

These tests verify that the `extract_pair` function correctly extracts
source‑target pairs from various input formats: tuples, lists, and dictionaries
with different key variants.
"""

import pytest
from src.domain.plugins.nrp_utils import extract_pair


def test_extract_pair_from_tuple_or_list():
    """Test extraction from tuple or list with two elements."""
    assert extract_pair(("R1", "R2")) == ("R1", "R2")
    assert extract_pair(["  R3 ", "R4  "]) == ("R3", "R4")


def test_extract_pair_from_dict_variants():
    """Test extraction from dictionaries using common key variants."""
    # Common key variants
    assert extract_pair({"req1": "R1", "req2": "R2"}) == ("R1", "R2")
    assert extract_pair({"from": "R1", "to": "R2"}) == ("R1", "R2")
    assert extract_pair({"r1": "R1", "r2": "R2"}) == ("R1", "R2")
    assert extract_pair({"source": "R1", "target": "R2"}) == ("R1", "R2")


def test_extract_pair_invalid_inputs():
    """Test that invalid inputs return empty strings."""
    assert extract_pair(None) == ("", "")
    assert extract_pair({}) == ("", "")
    assert extract_pair(["R1"]) == ("", "")   # single element, not a pair
    assert extract_pair("invalid_string") == ("", "")