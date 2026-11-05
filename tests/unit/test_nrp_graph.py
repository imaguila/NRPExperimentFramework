# Location: tests/test_graph.py
"""
Tests for graph generation and DAG validation (nrp_graph).

These tests verify that precedence graphs are correctly built and that
cycle detection works as expected.
"""

import networkx as nx
import pytest
from src.domain.plugins.nrp_graph import build_precedence_graph, validate_dag


def test_validate_dag_cyclic_model():
    """
    Test that a model with cyclic precedence relationships is correctly
    identified as non‑acyclic.

    The model has two items, R1 and R2, with cyclic dependencies:
        R1 -> R2 and R2 -> R1.
    The graph should contain both edges and fail the DAG validation.
    """
    from src.domain.core.item import DecisionItem
    from src.domain.core.model import OptimizationModel

    items = {
        "R1": DecisionItem("R1", "R1", {}),
        "R2": DecisionItem("R2", "R2", {}),
    }
    # Create a cycle: R1 -> R2 and R2 -> R1
    relationships = {
        "precedences": [
            {"from": "R1", "to": "R2"},
            {"from": "R2", "to": "R1"}
        ]
    }

    # Build the model using the dataclass constructor with named arguments
    model = OptimizationModel(
        name="CyclicModel",
        items=items,
        relationships=relationships
    )

    # Build the precedence graph
    G = build_precedence_graph(model)

    # Optional debug output (included for transparency)
    print("\n=== DEBUG: Constructed graph ===")
    print(f"Nodes: {G.nodes}")
    print(f"Edges: {G.edges}")
    print(f"Does it have edge R1->R2? {G.has_edge('R1', 'R2')}")
    print(f"Does it have edge R2->R1? {G.has_edge('R2', 'R1')}")

    # Verify that the graph contains the correct nodes and edges
    assert set(G.nodes) == {"R1", "R2"}
    assert G.has_edge("R1", "R2")
    assert G.has_edge("R2", "R1")

    # Verify that the graph is not a DAG (it should have a cycle)
    assert nx.is_directed_acyclic_graph(G) is False
    assert validate_dag(G) is False