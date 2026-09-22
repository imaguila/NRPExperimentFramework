# Location: src/solving/algorithms/simulated_annealing.py
"""
Simulated Annealing Multi‑Sweep Solver (Domain‑Agnostic).

This solver uses simulated annealing to explore the solution space for each
weight vector, performing a multi‑objective sweep to approximate the Pareto front.
"""

import random
import numpy as np
from typing import Dict, Any, List, Set

from src.domain.core.paretofront import ParetoFront
from src.domain.core.solution import Solution
from src.solving.core.solver import BaseSolver
from src.solving.core.problem import OptimizationProblem
from src.solving.core.evaluator import SolutionEvaluator
from src.solving.core.parameter import ParameterSpec
from src.solving.core.registry import register_solver
from src.solving.utils.graph import GraphUtils


@register_solver("Simulated Annealing (SA)")
class SimulatedAnnealingSolver(BaseSolver):
    """
    Simulated Annealing Multi‑Sweep Solver (Domain‑Agnostic).

    Performs a weight sweep, and for each weight vector runs a simulated
    annealing optimisation. The final set of solutions is filtered for
    Pareto dominance.
    """

    def __init__(
        self,
        iterations: int = 1000,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        weight_samples: int = 15
    ):
        super().__init__()
        self.iterations = iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.weight_samples = weight_samples

    @classmethod
    def get_parameter_schema(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="iterations",
                display_name="Iterations per Sweep",
                param_type=int,
                default=1000,
                min_value=100,
                max_value=20000,
                step=100,
                description="Number of cooling iterations per weight vector."
            ),
            ParameterSpec(
                name="initial_temp",
                display_name="Initial Temperature",
                param_type=float,
                default=100.0,
                min_value=1.0,
                max_value=1000.0,
                step=10.0,
                description="Initial temperature for accepting worsening moves."
            ),
            ParameterSpec(
                name="cooling_rate",
                display_name="Cooling Rate (Alpha)",
                param_type=float,
                default=0.95,
                min_value=0.50,
                max_value=0.999,
                step=0.01,
                description="Geometric cooling factor per iteration."
            ),
            ParameterSpec(
                name="weight_samples",
                display_name="Weight Sweeps",
                param_type=int,
                default=15,
                min_value=1,
                max_value=100,
                step=1,
                description="Number of multi‑objective weight vectors to explore."
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

        # Compute bounds using the evaluation API
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

        def validate_dependencies(selected: Set[str]) -> bool:
            for item in selected:
                prereqs = GraphUtils.get_transitive_prereqs(item, prereq_map)
                if not prereqs.issubset(selected):
                    return False
            return True

        def eval_scalar_score(candidate_set: Set[str], weights: np.ndarray) -> float:
            if not validate_dependencies(candidate_set):
                return -1e9
            if not evaluator.check_constraints(candidate_set, problem.constraints):
                return -1e9

            norm_score = 0.0
            for i, obj in enumerate(obj_keys):
                raw_val = evaluator.evaluate_attribute(candidate_set, obj)
                min_v, max_v = attr_bounds[obj]
                denom = (max_v - min_v) if (max_v - min_v) > 0 else 1.0
                scaled_val = (raw_val - min_v) / denom

                if not problem.is_maximize(obj):
                    scaled_val = 1.0 - scaled_val

                norm_score += weights[i] * scaled_val

            if cost_attr:
                cost_val = max(evaluator.evaluate_attribute(candidate_set, cost_attr), 0.0001)
                return norm_score / cost_val
            return norm_score

        for sweep_idx, weights in enumerate(weight_vectors):
            current_selected: Set[str] = set()
            current_score = eval_scalar_score(current_selected, weights)
            best_selected = set(current_selected)
            best_score = current_score

            temp = self.initial_temp

            for _ in range(self.iterations):
                if not active_items:
                    break

                target_item = random.choice(active_items)
                neighbor = set(current_selected)

                if target_item in neighbor:
                    neighbor.remove(target_item)
                else:
                    prereqs = GraphUtils.get_transitive_prereqs(target_item, prereq_map)
                    neighbor |= {target_item} | prereqs

                neighbor_score = eval_scalar_score(neighbor, weights)
                delta = neighbor_score - current_score

                if delta > 0 or random.random() < np.exp(delta / max(temp, 1e-6)):
                    current_selected = neighbor
                    current_score = neighbor_score

                    if current_score > best_score:
                        best_score = current_score
                        best_selected = set(current_selected)

                temp *= self.cooling_rate

            if best_selected:
                sol = evaluator.create_solution(
                    sol_id=f"SA_{sweep_idx+1}",
                    selected_ids=best_selected,
                    problem=problem
                )
                candidate_solutions.append(sol)

        if not candidate_solutions:
            return ParetoFront([])

        # Filter non‑dominated candidates using ParetoFront
        raw_front = ParetoFront(candidate_solutions)
        return raw_front.filter_non_dominated(problem.get_objective_directions())