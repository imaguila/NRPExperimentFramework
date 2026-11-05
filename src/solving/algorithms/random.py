# Location: src/solving/algorithms/random.py
"""
Random Search Solver with dependency repair and Pareto filtering.

This solver generates random subsets of items, repairs them by adding
transitive prerequisites, and filters for feasibility before constructing
solutions. The final set is filtered for Pareto dominance.
"""

import random
from typing import Dict, Any, List, Set

from src.domain.core.paretofront import ParetoFront
from src.solving.core.solver import BaseSolver
from src.solving.core.problem import OptimizationProblem
from src.solving.core.evaluator import SolutionEvaluator
from src.solving.core.parameter import ParameterSpec
from src.solving.core.registry import register_solver
from src.solving.utils.graph import GraphUtils


@register_solver("Random Search / Sampling")
class RandomSearchSolver(BaseSolver):
    """
    Random Sampling Solver with transitive dependency repair.
    """

    def __init__(self, num_samples: int = 1000):
        super().__init__()
        self.num_samples = num_samples

    @classmethod
    def get_parameter_schema(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="num_samples",
                display_name="Number of Samples",
                param_type=int,
                default=1000,
                min_value=50,
                max_value=50000,
                step=100,
                description="Number of random configurations to sample."
            )
        ]

    def solve(self, problem: OptimizationProblem, **kwargs) -> ParetoFront:
        evaluator = SolutionEvaluator(problem.model)

        # Extract dependencies
        relationships = getattr(problem.model, "relationships", {}) or {}
        precedences = (
            relationships.get("precedences", []) or
            getattr(problem.model, "precedences", []) or
            getattr(problem.model, "dependencies", [])
        )
        couplings = relationships.get("couplings", [])
        prereq_map, _ = GraphUtils.build_dependency_maps(precedences, couplings)

        active_items = [str(item_id) for item_id in problem.model.items.keys()]

        if not active_items or not problem.objectives:
            return ParetoFront([])

        candidates = []
        seen_sets = set()

        for _ in range(self.num_samples):
            # 1. Random sampling
            k = random.randint(1, len(active_items))
            raw_sample = set(random.sample(active_items, k))

            # 2. Repair transitive dependencies
            candidate_set = set(raw_sample)
            for item_id in raw_sample:
                prereqs = GraphUtils.get_transitive_prereqs(item_id, prereq_map)
                candidate_set |= prereqs

            # 3. Check feasibility and duplicates
            fkey = frozenset(candidate_set)
            if fkey not in seen_sets:
                seen_sets.add(fkey)
                if evaluator.check_constraints(candidate_set, problem.constraints):
                    sol = evaluator.create_solution(
                        sol_id=f"RND_{len(candidates)+1}",
                        selected_ids=candidate_set,
                        problem=problem
                    )
                    candidates.append(sol)

        # 4. Filter and return Pareto front
        raw_front = ParetoFront(candidates)
        return raw_front.filter_non_dominated(problem.get_objective_directions())