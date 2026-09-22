# Location: src/solving/algorithms/milp.py
"""
Mixed Integer Linear Programming (MILP) Solver with Multi‑Objective Sweep.

This solver uses SciPy's MILP implementation to solve the optimisation problem
exactly for each weight vector, generating a set of candidate solutions that
are later filtered for Pareto dominance.
"""

from typing import Dict, Any, List, Set, Optional
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from src.domain.core.paretofront import ParetoFront
from src.domain.core.solution import Solution
from src.solving.core.solver import BaseSolver
from src.solving.core.problem import OptimizationProblem
from src.solving.core.evaluator import SolutionEvaluator
from src.solving.core.parameter import ParameterSpec
from src.solving.core.registry import register_solver


@register_solver("Mixed Integer Linear Programming (MILP Exact)")
class MILPSolver(BaseSolver):
    """
    Exact MILP solver using SciPy's `milp` function.

    Performs a weight‑sweep to approximate the Pareto front by solving
    the scalarised problem for a set of weight vectors. Each MILP solve
    is subject to a time limit.
    """

    def __init__(self, weight_samples: int = 25, time_limit_per_sweep: float = 3.0):
        super().__init__()
        self.weight_samples = weight_samples
        self.time_limit_per_sweep = time_limit_per_sweep

    @classmethod
    def get_parameter_schema(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="weight_samples",
                display_name="Weight Sweeps",
                param_type=int,
                default=25,
                min_value=5,
                max_value=200,
                step=5,
                description="Number of weight combinations used for the sweep."
            ),
            ParameterSpec(
                name="time_limit_per_sweep",
                display_name="Timeout per Sweep (s)",
                param_type=float,
                default=3.0,
                min_value=0.5,
                max_value=30.0,
                step=0.5,
                description="Maximum solve time (seconds) per weight sweep."
            ),
        ]

    def solve(self, problem: OptimizationProblem, **kwargs) -> ParetoFront:
        evaluator = SolutionEvaluator(problem.model)

        active_ids = [str(k) for k in problem.model.items.keys()]
        n = len(active_ids)
        if n == 0 or not problem.objectives:
            return ParetoFront([])

        id_to_idx = {item_id: idx for idx, item_id in enumerate(active_ids)}
        obj_keys = list(problem.objectives.keys())
        num_objs = len(obj_keys)

        # 1. Compute attribute coefficients per item
        all_attrs = set(problem.objectives.keys()).union(problem.constraints.keys())
        attr_vectors: Dict[str, np.ndarray] = {}
        for attr in all_attrs:
            vec = np.zeros(n)
            for j, item_id in enumerate(active_ids):
                vec[j] = evaluator.evaluate_attribute({item_id}, attr)
            attr_vectors[attr] = vec

        # 2. Linear constraints
        constraints_list: List[LinearConstraint] = []

        # 2a. Attribute threshold constraints
        for attr, spec in problem.constraints.items():
            vec = attr_vectors[attr]
            target = getattr(spec, "value", spec)
            op = getattr(spec, "operator", None)
            op_str = str(op.value if hasattr(op, "value") else op).lower()

            if "less" in op_str or op_str in ["<=", "<", "le"]:
                constraints_list.append(LinearConstraint(vec, -np.inf, float(target)))
            elif "greater" in op_str or op_str in [">=", ">", "ge"]:
                constraints_list.append(LinearConstraint(vec, float(target), np.inf))
            elif "equal" in op_str or op_str in ["==", "="]:
                constraints_list.append(LinearConstraint(vec, float(target), float(target)))
            elif isinstance(target, (int, float)):
                constraints_list.append(LinearConstraint(vec, -np.inf, float(target)))

        # 2b. Dependency linear constraints (x_tgt - x_src <= 0)
        relationships = getattr(problem.model, "relationships", {}) or {}
        dependencies = (
            relationships.get("precedences", []) or
            getattr(problem.model, "dependencies", None) or
            getattr(problem.model, "precedences", None) or []
        )

        for dep in dependencies:
            if isinstance(dep, dict):
                src = str(dep.get("source") or dep.get("prerequisite") or dep.get("from"))
                tgt = str(dep.get("target") or dep.get("dependent") or dep.get("to"))
            elif isinstance(dep, (list, tuple)) and len(dep) >= 2:
                src, tgt = str(dep[0]), str(dep[1])
            else:
                continue

            if src in id_to_idx and tgt in id_to_idx:
                src_idx = id_to_idx[src]
                tgt_idx = id_to_idx[tgt]
                dep_row = np.zeros(n)
                dep_row[tgt_idx] = 1.0
                dep_row[src_idx] = -1.0
                constraints_list.append(LinearConstraint(dep_row, -np.inf, 0.0))

        bounds = Bounds(0.0, 1.0)
        integrality = np.ones(n)

        weight_vectors = self._generate_weight_vectors(num_objs, self.weight_samples)
        candidate_solutions: List[Solution] = []

        for w_vec in weight_vectors:
            c = np.zeros(n)
            for idx, obj in enumerate(obj_keys):
                # SciPy minimises by default; invert sign for maximisation objectives
                factor = -1.0 if problem.is_maximize(obj) else 1.0
                c += w_vec[idx] * factor * attr_vectors[obj]

            # Run SciPy MILP with time limit per sweep
            res = milp(
                c=c,
                integrality=integrality,
                bounds=bounds,
                constraints=constraints_list,
                options={"time_limit": self.time_limit_per_sweep}
            )

            if res.success and res.x is not None:
                selected_indices = np.where(res.x >= 0.5)[0]
                selected_ids = {active_ids[j] for j in selected_indices}

                if selected_ids:
                    sol = evaluator.create_solution(
                        sol_id=f"MILP_{len(candidate_solutions)+1}",
                        selected_ids=selected_ids,
                        problem=problem
                    )
                    candidate_solutions.append(sol)

        if not candidate_solutions:
            return ParetoFront([])

        # Filter non‑dominated candidates using ParetoFront
        raw_front = ParetoFront(candidate_solutions)
        return raw_front.filter_non_dominated(problem.get_objective_directions())

    @staticmethod
    def _generate_weight_vectors(num_objs: int, samples: int) -> List[List[float]]:
        """
        Generate a set of weight vectors for multi‑objective scalarisation.

        For 1 objective, returns [1.0]. For 2 objectives, uniformly spaced weights.
        For more, random Dirichlet samples are returned.

        Args:
            num_objs: Number of objectives.
            samples: Number of weight vectors to generate.

        Returns:
            List of weight vectors (each a list of floats).
        """
        if num_objs <= 1:
            return [[1.0]]
        if num_objs == 2:
            return [[w, 1.0 - w] for w in np.linspace(0.0, 1.0, samples)]

        raw_samples = np.random.dirichlet(np.ones(num_objs), size=samples)
        return raw_samples.tolist()