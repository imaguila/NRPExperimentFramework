# Location: src/solving/utils/graph.py
"""
Graph utilities for dependency management and solution repair.

This module provides helper functions for building and traversing
precedence and coupling graphs, computing transitive dependencies,
and repairing infeasible solutions by adding required prerequisites
or removing items that violate constraints.
"""

from typing import Dict, Set, Tuple, List, Any, Optional
from src.solving.core.evaluator import SolutionEvaluator


class GraphUtils:
    """
    Utilities for precedence graphs, couplings, and solution repair.

    Provides static methods to build dependency maps (forward and backward),
    compute transitive prerequisites and dependents, and repair candidate
    solutions by enforcing dependencies and constraint feasibility.
    """

    @staticmethod
    def build_dependency_maps(
        precedences: List[Any],
        couplings: Optional[List[Any]] = None
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Build prerequisite and dependent maps integrating both unidirectional
        precedences and bidirectional couplings.

        Args:
            precedences: List of precedence pairs (A -> B means A must precede B).
            couplings: List of coupling pairs (A <-> B means both depend on each other).

        Returns:
            A tuple (prereq_map, dependent_map) where:
                - prereq_map[dst] = set of sources that must be selected before dst.
                - dependent_map[src] = set of targets that depend on src.
        """
        prereq_map: Dict[str, Set[str]] = {}
        dependent_map: Dict[str, Set[str]] = {}

        # 1. Unidirectional precedences (A -> B: B requires A)
        for p in precedences or []:
            if isinstance(p, (tuple, list)) and len(p) >= 2:
                src, dst = str(p[0]), str(p[1])
            elif isinstance(p, dict):
                src = str(p.get("from") or p.get("r1") or p.get("source") or p.get("prerequisite", ""))
                dst = str(p.get("to") or p.get("r2") or p.get("target") or p.get("dependent", ""))
            else:
                continue

            if src and dst:
                prereq_map.setdefault(dst, set()).add(src)
                dependent_map.setdefault(src, set()).add(dst)

        # 2. Bidirectional couplings (A <-> B: mutually required)
        for c in couplings or []:
            if isinstance(c, (tuple, list)) and len(c) >= 2:
                n1, n2 = str(c[0]), str(c[1])
            elif isinstance(c, dict):
                n1 = str(c.get("r1", ""))
                n2 = str(c.get("r2", ""))
            else:
                continue

            if n1 and n2:
                prereq_map.setdefault(n1, set()).add(n2)
                prereq_map.setdefault(n2, set()).add(n1)
                dependent_map.setdefault(n1, set()).add(n2)
                dependent_map.setdefault(n2, set()).add(n1)

        return prereq_map, dependent_map

    @staticmethod
    def get_transitive_prereqs(node_id: str, prereq_map: Dict[str, Set[str]]) -> Set[str]:
        """
        Compute all transitive prerequisites (backward closure) for a given node.

        Args:
            node_id: The starting node.
            prereq_map: Prerequisite map (dst -> set of sources).

        Returns:
            A set of all nodes that must be selected before the given node
            (direct and indirect).
        """
        visited: Set[str] = set()
        stack = [node_id]
        while stack:
            curr = stack.pop()
            for p in prereq_map.get(curr, set()):
                if p not in visited:
                    visited.add(p)
                    stack.append(p)
        return visited

    @staticmethod
    def get_transitive_dependents(node_id: str, dependent_map: Dict[str, Set[str]]) -> Set[str]:
        """
        Compute all transitive dependents (forward closure) for a given node.

        Args:
            node_id: The starting node.
            dependent_map: Dependent map (src -> set of targets).

        Returns:
            A set of all nodes that depend (directly or indirectly) on the given node.
        """
        visited: Set[str] = set()
        stack = [node_id]
        while stack:
            curr = stack.pop()
            for d in dependent_map.get(curr, set()):
                if d not in visited:
                    visited.add(d)
                    stack.append(d)
        return visited

    @staticmethod
    def repair_solution(
        selected_ids: Set[str],
        prereq_map: Dict[str, Set[str]],
        dependent_map: Dict[str, Set[str]],
        constraints: Dict[str, Any],
        evaluator: SolutionEvaluator
    ) -> Set[str]:
        """
        Repair a candidate solution by adding missing prerequisites (forward repair)
        and removing items that violate constraints or have excluded dependents
        (backward repair).

        The repair process consists of two phases:
            1. Forward repair: add all transitive prerequisites for selected items.
            2. Backward repair: while the solution is infeasible, remove an item
               and all its dependents, then re‑check constraints.

        Args:
            selected_ids: Set of initially selected item IDs.
            prereq_map: Prerequisite map (dst -> sources).
            dependent_map: Dependent map (src -> targets).
            constraints: Constraint specifications for feasibility checks.
            evaluator: SolutionEvaluator used to check constraints.

        Returns:
            A repaired set of item IDs that is feasible and respects dependencies.
        """
        repaired_ids = set(selected_ids)

        # 1. Forward Repair: include all required prerequisites
        for rid in list(repaired_ids):
            prereqs = GraphUtils.get_transitive_prereqs(rid, prereq_map)
            repaired_ids.update(prereqs)

        # 2. Backward Repair: remove items if constraints are violated
        while not evaluator.check_constraints(repaired_ids, constraints) and repaired_ids:
            # Remove the first item and all its dependents
            candidate_to_drop = list(repaired_ids)[0]
            dependents = GraphUtils.get_transitive_dependents(candidate_to_drop, dependent_map)
            to_remove = dependents | {candidate_to_drop}
            repaired_ids -= to_remove

        return repaired_ids