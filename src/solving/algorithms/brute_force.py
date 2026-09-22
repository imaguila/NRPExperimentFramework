# Location: src/solving/algorithms/brute_force.py
"""
Exact Brute Force Solver with branch‑and‑bound pruning, exploration limits,
and dynamic multi‑objective Pareto filtering.
"""

from typing import Dict, Any, List, Set

from src.domain.core.paretofront import ParetoFront
from src.domain.core.solution import Solution
from src.solving.core.solver import BaseSolver
from src.solving.core.problem import OptimizationProblem
from src.solving.core.evaluator import SolutionEvaluator
from src.solving.core.parameter import ParameterSpec
from src.solving.core.registry import register_solver
from src.solving.utils.graph import GraphUtils


@register_solver("Exact Brute Force (Pruned)")
class BruteForceSolver(BaseSolver):
    """
    Exact Brute Force Solver with branch‑and‑bound pruning,
    exploration limits, and dynamic multi‑objective Pareto filtering.
    """

    def __init__(self, max_evaluations: int = 30000):
        super().__init__()
        self.max_evaluations = max_evaluations

    @classmethod
    def get_parameter_schema(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="max_evaluations",
                display_name="Max Evaluation Limit",
                param_type=int,
                default=30000,
                min_value=1000,
                max_value=500000,
                step=5000,
                description="Maximum number of explored nodes to prevent CPU lock."
            )
        ]

    def solve(self, problem: OptimizationProblem, **kwargs) -> ParetoFront:
        evaluator = SolutionEvaluator(problem.model)

        # Build dependency and coupling maps
        relationships = getattr(problem.model, "relationships", {}) or {}
        precedences = (
            relationships.get("precedences", []) or
            getattr(problem.model, "precedences", []) or
            getattr(problem.model, "dependencies", [])
        )
        couplings = relationships.get("couplings", [])
        prereq_map, _ = GraphUtils.build_dependency_maps(precedences, couplings)

        active_ids = [str(k) for k in problem.model.items.keys()]
        n = len(active_ids)
        if n == 0 or not problem.objectives:
            return ParetoFront([])

        # Protection: Brute force is O(2^N). If N > 22, cap the items to explore.
        if n > 22:
           # self.logger.warning(
            #    "Brute Force only supports up to 22 decision items. "
            #    "Execution skipped."
            #) 
            #  In order to prevent the solver from running indefinitely, we will return an empty Pareto front if the number of items exceeds 22.
            return ParetoFront([])


        # Extract upper bounds from constraints for early branch pruning
        upper_limits = {}
        for attr, spec in problem.constraints.items():
            op = getattr(spec, "operator", None)
            op_str = str(op.value if hasattr(op, "value") else op).lower()
            val = getattr(spec, "value", spec)
            if ("less" in op_str or op_str in ["<=", "<", "le"]) and isinstance(val, (int, float)):
                upper_limits[attr] = float(val)

        candidates: List[Solution] = []
        node_count = 0

        def backtrack(index: int, current_selected: List[str]):
            nonlocal node_count
            node_count += 1

            if node_count >= self.max_evaluations:
                return

            selected_set = set(current_selected)

            # Prune if any cumulative constraint limit is exceeded
            for attr, max_val in upper_limits.items():
                if evaluator.evaluate_attribute(selected_set, attr) > max_val:
                    return

            # Leaf of the search tree
            if index == n:
                if self._is_feasible(selected_set, problem, evaluator, prereq_map):
                    sol = evaluator.create_solution(
                        sol_id=f"BF_{len(candidates) + 1}",
                        selected_ids=selected_set,
                        problem=problem
                    )
                    candidates.append(sol)
                return

            # Branch 1: Include the current item
            backtrack(index + 1, current_selected + [active_ids[index]])

            # Branch 2: Exclude the current item
            if node_count < self.max_evaluations:
                backtrack(index + 1, current_selected)

        backtrack(0, [])

        if not candidates:
            return ParetoFront([])

        # Filter non‑dominated candidates using ParetoFront
        raw_front = ParetoFront(candidates)
        return raw_front.filter_non_dominated(problem.get_objective_directions())

    def _is_feasible(
        self,
        selected_ids: Set[str],
        problem: OptimizationProblem,
        evaluator: SolutionEvaluator,
        prereq_map: Dict[str, Set[str]]
    ) -> bool:
        """Check graph prerequisites and threshold constraints."""
        for item in selected_ids:
            prereqs = GraphUtils.get_transitive_prereqs(item, prereq_map)
            if not prereqs.issubset(selected_ids):
                return False

        return evaluator.check_constraints(selected_ids, problem.constraints)