# Location: src/domain/plugins/nrp_graph.py
"""
NRP Precedence Graph utilities.

This module provides functions to build a directed graph from the
precedence relationships in an NRP model and to validate that the graph
is acyclic (DAG).
"""

import networkx as nx
from src.domain.core.model import OptimizationModel
from src.domain.plugins.nrp_utils import extract_pair


def build_precedence_graph(model: OptimizationModel) -> nx.DiGraph:
    """
    Build a directed graph from the precedence relationships in the model.

    Nodes represent items (requirements), and directed edges represent
    precedence constraints (source must be selected before target).

    Args:
        model: The OptimizationModel containing items and precedences.

    Returns:
        A NetworkX DiGraph with nodes labelled by item descriptions.
    """
    G = nx.DiGraph()

    # 1. Add all nodes
    for item_id, item_obj in model.items.items():
        str_id = str(item_id).strip()
        G.add_node(str_id, label=getattr(item_obj, "description", str_id))

    # 2. Extract precedences and add edges
    precedences = model.relationships.get("precedences", [])
    print(f"\n=== build_precedence_graph: Processing {len(precedences)} precedences ===")

    for i, rule in enumerate(precedences):
        pred, succ = extract_pair(rule)
        print(f"  Precedence {i+1}: rule={rule} -> pred='{pred}', succ='{succ}'")

        if pred and succ and pred in G and succ in G:
            G.add_edge(pred, succ)
            print(f"    ✅ Edge added: {pred} -> {succ}")
        else:
            print(f"    ❌ Edge NOT added: {pred} -> {succ} (node(s) do not exist)")

    return G


def validate_dag(G: nx.DiGraph) -> bool:
    """
    Check whether the directed graph is a DAG (Directed Acyclic Graph).

    Args:
        G: A NetworkX DiGraph.

    Returns:
        True if the graph is acyclic, False otherwise.
    """
    try:
        return nx.is_directed_acyclic_graph(G)
    except Exception:
        return False