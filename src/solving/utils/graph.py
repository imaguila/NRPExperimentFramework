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
        couplings: Optional[List[Any]] = None,
    ) -> Tuple[
        Dict[str, Set[str]],
        Dict[str, Set[str]],
    ]:
        """
        Build prerequisite and dependent maps.

        Implication convention
        ----------------------
        A precedence entry:

            (req1, req2)

        means:

            selecting req1 requires req2

        Therefore:

            prereq_map[req1] contains req2
            dependent_map[req2] contains req1
        """

        prereq_map: Dict[str, Set[str]] = {}
        dependent_map: Dict[str, Set[str]] = {}

        # ---------------------------------------------------------
        # Unidirectional precedences
        # ---------------------------------------------------------

        for precedence in precedences or []:

            if (
                isinstance(precedence, (tuple, list))
                and len(precedence) >= 2
            ):
                src = str(precedence[0]).strip()
                tgt = str(precedence[1]).strip()

            elif isinstance(precedence, dict):
                src = str(
                    precedence.get("from")
                    or precedence.get("req1")
                    or precedence.get("r1")
                    or precedence.get("source")
                    or precedence.get("dependent")
                    or ""
                ).strip()

                tgt = str(
                    precedence.get("to")
                    or precedence.get("req2")
                    or precedence.get("r2")
                    or precedence.get("target")
                    or precedence.get("prerequisite")
                    or ""
                ).strip()

            else:
                continue

            if not src or not tgt:
                continue

            # selecting src requires tgt

            prereq_map.setdefault(
                src,
                set(),
            ).add(tgt)

            dependent_map.setdefault(
                tgt,
                set(),
            ).add(src)

        # ---------------------------------------------------------
        # Couplings
        # ---------------------------------------------------------

        for coupling in couplings or []:

            if (
                isinstance(coupling, (tuple, list))
                and len(coupling) >= 2
            ):
                first = str(coupling[0]).strip()
                second = str(coupling[1]).strip()

            elif isinstance(coupling, dict):
                first = str(
                    coupling.get("req1")
                    or coupling.get("r1")
                    or coupling.get("from")
                    or ""
                ).strip()

                second = str(
                    coupling.get("req2")
                    or coupling.get("r2")
                    or coupling.get("to")
                    or ""
                ).strip()

            else:
                continue

            if not first or not second:
                continue

            prereq_map.setdefault(
                first,
                set(),
            ).add(second)

            prereq_map.setdefault(
                second,
                set(),
            ).add(first)

            dependent_map.setdefault(
                first,
                set(),
            ).add(second)

            dependent_map.setdefault(
                second,
                set(),
            ).add(first)

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