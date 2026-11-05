# tests/test_solving.py
"""
Mock model for solver tests.

This module provides a synthetic domain‑agnostic model used as a fixture
for testing optimisation solvers. It contains a small set of decision
items with cost and value attributes, and a simple precedence relationship.
"""


class MockModel:
    """
    Synthetic agnostic model with items, attributes, and relationships.

    This mock model is used for solver tests. It provides a small set of
    decision items (A, B, C, D) with cost and value attributes, and a
    precedence relationship (A must precede B). The model is intentionally
    simple to allow fast solver execution during tests.
    """
    def __init__(self):
        from src.domain.core.item import DecisionItem
        self.name = "MockModel"
        self.items = {
            "A": DecisionItem("A", "Item A", {"cost": 10.0, "value": 50.0}),
            "B": DecisionItem("B", "Item B", {"cost": 20.0, "value": 80.0}),
            "C": DecisionItem("C", "Item C", {"cost": 15.0, "value": 40.0}),
            "D": DecisionItem("D", "Item D", {"cost": 30.0, "value": 100.0}),
        }
        self.relationships = {
            "precedences": [
                {"req1": "A", "req2": "B"}  # B depends on A
            ],
            "couplings": []
        }
        self.attribute_definitions = {}