# Location: src/solving/algorithms/greedy.py
"""
Greedy Multi‑Weight Sweep Solver (Domain‑Agnostic).

This solver constructs solutions by iteratively adding the best item according
to a weighted scalarisation of the objectives, sweeping over multiple weight
vectors to explore the Pareto front.
"""

from typing import Dict, Any, List, Set
import numpy as np

from src.domain.core.paretofront import ParetoFront
from src.domain.core.solution import Solution
from src.solving.core.solver import BaseSolver
from src.solving.core.problem import OptimizationProblem
from src.solving.core.evaluator import SolutionEvaluator
from src.solving.core.parameter import ParameterSpec
from src.solving.core.registry import register_solver
from src.solving.utils.graph import GraphUtils


@register_solver("Greedy Multi-Weight Sweep (Constructive)")
class GreedySolver(BaseSolver):
    """
    Greedy Multi‑Weight Sweep Solver (Domain‑Agnostic).

    Uses Min‑Max normalisation and explores the Pareto front by sweeping
    over multiple weight vectors. At each step, it adds the item that
    yields the best scalarised score while respecting constraints and
    dependencies.
    """

    def __init__(self, weight_samples: int = 30):
        super().__init__()
        self.weight_samples = weight_samples

    @classmethod
    def get_parameter_schema(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="weight_samples",
                display_name="Weight Samples (Sweeps)",
                param_type=int,
                default=30,
                min_value=5,
                max_value=500,
                step=5,
                description="Number of weight combinations used for the multi‑objective sweep."
            )
        ]

    def _generate_weight_vectors(self, num_objectives: int) -> np.ndarray:
        """
        Generate a set of weight vectors for multi‑objective scalarisation.

        For 2 objectives, evenly spaced weights are used.
        For more than 2, random Dirichlet samples plus the extreme points
        (one‑hot vectors) are generated.

        Args:
            num_objectives: Number of objectives.

        Returns:
            An array of weight vectors (samples × num_objectives).
        """
        if num_objectives <= 1:
            return np.array([[1.0]])
        elif num_objectives == 2:
            w1 = np.linspace(0.0, 1.0, self.weight_samples)
            return np.column_stack([w1, 1.0 - w1])
        else:
            np.random.seed(42)
            weights = np.random.dirichlet(np.ones(num_objectives), size=self.weight_samples)
            return np.vstack([weights, np.eye(num_objectives)])

    def solve(self, problem: OptimizationProblem, **kwargs) -> ParetoFront:
        evaluator = SolutionEvaluator(problem.model)

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

        obj_keys = list(problem.objectives.keys())
        num_objs = len(obj_keys)

        # Initial approximate bounds for normalisation
        attr_bounds = {}
        for obj in obj_keys:
            vals = [evaluator.evaluate_attribute({rid}, obj) for rid in active_items]
            min_v = min(vals) if vals else 0.0
            max_v = max(vals) if vals else 1.0
            attr_bounds[obj] = (min_v, max_v if max_v > min_v else min_v + 1.0)

        weight_vectors = self._generate_weight_vectors(num_objs)
        candidate_solutions: List[Solution] = []

        cost_attr = getattr(problem, "primary_constraint_attr", None)
        if not cost_attr and problem.constraints:
            cost_attr = list(problem.constraints.keys())[0]

        for sweep_idx, weights in enumerate(weight_vectors):
            selected_ids: Set[str] = set()

            while True:
                best_item = None
                best_score = -float("inf")

                for rid in active_items:
                    if rid in selected_ids:
                        continue

                    prereqs = GraphUtils.get_transitive_prereqs(rid, prereq_map)
                    candidate_set = selected_ids | {rid} | prereqs

                    if not evaluator.check_constraints(candidate_set, problem.constraints):
                        continue

                    # Compute normalised scalarised score
                    norm_score = 0.0
                    for i, obj in enumerate(obj_keys):
                        raw_val = evaluator.evaluate_attribute(candidate_set, obj)
                        min_v, max_v = attr_bounds[obj]
                        scaled_val = raw_val / (max_v * len(candidate_set) if max_v > 0 else 1.0)

                        if not problem.is_maximize(obj):
                            scaled_val = 1.0 - scaled_val

                        norm_score += weights[i] * scaled_val

                    # Optionally divide by cost (if a primary constraint attribute is set)
                    if cost_attr:
                        cost_val = max(evaluator.evaluate_attribute(candidate_set, cost_attr), 0.0001)
                        score = norm_score / cost_val
                    else:
                        score = norm_score

                    if score > best_score:
                        best_score = score
                        best_item = (rid, candidate_set)

                if best_item is None:
                    break

                selected_ids = best_item[1]

            if selected_ids:
                sol = evaluator.create_solution(f"G_{sweep_idx+1}", selected_ids, problem)
                candidate_solutions.append(sol)

        if not candidate_solutions:
            return ParetoFront([])

        # Filter non‑dominated candidates using ParetoFront
        raw_front = ParetoFront(candidate_solutions)
        return raw_front.filter_non_dominated(problem.get_objective_directions())