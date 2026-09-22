# Location: src/solving/algorithms/nsga2.py
"""
NSGA‑II Multi‑Objective Genetic Algorithm.

This solver implements the NSGA‑II algorithm with fast non‑dominated sorting
and crowding distance selection for multi‑objective optimisation.
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


@register_solver("NSGA-II Genetic Algorithm")
class NSGA2Solver(BaseSolver):
    """
    NSGA‑II Multi‑Objective Genetic Algorithm.

    Uses fast non‑dominated sorting and crowding distance for selection.
    """

    def __init__(
        self,
        pop_size: int = 50,
        generations: int = 60,
        cx_prob: float = 0.8,
        mut_prob: float = 0.15
    ):
        super().__init__()
        self.pop_size = pop_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob

    @classmethod
    def get_parameter_schema(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="pop_size",
                display_name="Population Size",
                param_type=int,
                default=50,
                min_value=10,
                max_value=500,
                step=10,
                description="Size of the population of solutions."
            ),
            ParameterSpec(
                name="generations",
                display_name="Generations",
                param_type=int,
                default=60,
                min_value=5,
                max_value=1000,
                step=5,
                description="Number of evolutionary generations."
            ),
            ParameterSpec(
                name="cx_prob",
                display_name="Crossover Probability",
                param_type=float,
                default=0.8,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                description="Probability of crossover between pairs of parents."
            ),
            ParameterSpec(
                name="mut_prob",
                display_name="Mutation Probability",
                param_type=float,
                default=0.15,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                description="Probability of mutation per gene."
            )
        ]

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

        num_items = len(active_items)
        obj_keys = list(problem.objectives.keys())

        def decode_and_repair(chromosome: List[bool]) -> Set[str]:
            selected = {active_items[i] for i, b in enumerate(chromosome) if b}
            expanded = set(selected)
            for item in selected:
                prereqs = GraphUtils.get_transitive_prereqs(item, prereq_map)
                expanded |= prereqs

            if not evaluator.check_constraints(expanded, problem.constraints):
                return set()
            return expanded

        def evaluate_fitness(chromosome: List[bool]) -> np.ndarray:
            selected_ids = decode_and_repair(chromosome)
            if not selected_ids and any(chromosome):
                # Penalty for infeasible solution: all objectives set to +infinity
                return np.array([float("inf") for _ in obj_keys])

            fit = []
            for k in obj_keys:
                val = evaluator.evaluate_attribute(selected_ids, k)
                # Internal conversion to minimisation: negate if maximising
                if problem.is_maximize(k):
                    fit.append(-val)
                else:
                    fit.append(val)
            return np.array(fit)

        # Initial population
        population = [[random.random() < 0.3 for _ in range(num_items)] for _ in range(self.pop_size)]

        for gen in range(self.generations):
            offspring = []
            while len(offspring) < self.pop_size:
                p1, p2 = random.sample(population, 2)
                c1, c2 = list(p1), list(p2)

                if random.random() < self.cx_prob:
                    pt = random.randint(1, max(1, num_items - 1))
                    c1 = p1[:pt] + p2[pt:]
                    c2 = p2[:pt] + p1[pt:]

                for c in [c1, c2]:
                    for i in range(num_items):
                        if random.random() < self.mut_prob:
                            c[i] = not c[i]
                    offspring.append(c)

            combined = population + offspring
            fits = np.array([evaluate_fitness(ind) for ind in combined])

            # NSGA-II selection: Non-dominated Sort + Crowding Distance
            fronts = self._fast_non_dominated_sort(fits)
            new_pop = []

            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend([combined[i] for i in front])
                else:
                    # Fill the last front by sorting by crowding distance
                    distances = self._calculate_crowding_distance(fits[front])
                    sorted_front = [front[i] for i in np.argsort(-distances)]
                    needed = self.pop_size - len(new_pop)
                    new_pop.extend([combined[i] for i in sorted_front[:needed]])
                    break

            population = new_pop

        # Build final solutions
        candidate_solutions: List[Solution] = []
        seen_sets = set()

        for chrom in population:
            selected_ids = decode_and_repair(chrom)
            if selected_ids:
                fkey = frozenset(selected_ids)
                if fkey not in seen_sets:
                    seen_sets.add(fkey)
                    sol = evaluator.create_solution(
                        sol_id=f"NSGA2_{len(candidate_solutions)+1}",
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
    def _fast_non_dominated_sort(fits: np.ndarray) -> List[List[int]]:
        """Assumes all objectives have been converted to minimisation."""
        n = len(fits)
        domination_sets = [[] for _ in range(n)]
        dominated_counts = np.zeros(n, dtype=int)
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(fits[i] <= fits[j]) and np.any(fits[i] < fits[j]):
                    domination_sets[i].append(j)
                elif np.all(fits[j] <= fits[i]) and np.any(fits[j] < fits[i]):
                    dominated_counts[i] += 1

            if dominated_counts[i] == 0:
                fronts[0].append(i)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in domination_sets[p]:
                    dominated_counts[q] -= 1
                    if dominated_counts[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return [f for f in fronts if len(f) > 0]

    @staticmethod
    def _calculate_crowding_distance(fits: np.ndarray) -> np.ndarray:
        n, num_objs = fits.shape
        distances = np.zeros(n, dtype=float)

        if n <= 2:
            return np.full(n, np.inf)

        for m in range(num_objs):
            values = fits[:, m]
            finite_idx = np.flatnonzero(np.isfinite(values))

            if len(finite_idx) <= 2:
                distances[finite_idx] = np.inf
                continue

            order = finite_idx[np.argsort(values[finite_idx])]
            min_value = values[order[0]]
            max_value = values[order[-1]]
            norm_range = max_value - min_value

            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf

            if not np.isfinite(norm_range) or norm_range <= 0:
                continue

            for i in range(1, len(order) - 1):
                if np.isinf(distances[order[i]]):
                    continue

                previous_value = values[order[i - 1]]
                next_value = values[order[i + 1]]

                distances[order[i]] += (
                    next_value - previous_value
                ) / norm_range

        return distances