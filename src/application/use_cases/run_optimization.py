# src/application/use_cases/run_optimization.py

# Location: src/application/use_cases/run_optimization.py

"""
Headless optimisation execution use case.

This module executes an already configured solver against an
OptimizationProblem and returns a structured result. It contains no
Streamlit dependencies and can therefore be reused by the web interface,
tests, notebooks, scripts, or future command-line clients.
"""

from __future__ import annotations


import time
from typing import Callable, Dict, List, Sequence

from src.application.dto import (
    CompositeOptimizationRunResult,
    OptimizationRunResult,
)
from src.domain.core.model import OptimizationModel
from src.domain.core.paretofront import ParetoFront
from src.domain.core.solution import Solution
from src.solving.core.problem import OptimizationProblem
from src.solving.core.solver import BaseSolver


def run_optimization(
    problem: OptimizationProblem,
    solver: BaseSolver,
) -> OptimizationRunResult:
    """
    Execute a solver against an optimisation problem.

    Parameters
    ----------
    problem:
        Fully configured optimisation problem containing the selected
        model, objectives, directions, and constraints.

    solver:
        Configured solver instance.

    Returns
    -------
    OptimizationRunResult
        Structured execution result containing the problem, solver,
        Pareto front, DataFrame representation, and execution time.
    """
    if problem is None:
        raise ValueError(
            "An OptimizationProblem is required to run optimisation."
        )

    if solver is None:
        raise ValueError(
            "A configured solver is required to run optimisation."
        )

    started_at = time.perf_counter()

    pareto_front = solver.solve(problem)

    execution_time = time.perf_counter() - started_at

    pareto_df = pareto_front.to_dataframe()

    return OptimizationRunResult(
        problem=problem,
        solver=solver,
        pareto_front=pareto_front,
        pareto_df=pareto_df,
        execution_time=execution_time,
    )

def _copy_solution_with_branch_provenance(
    solution: Solution,
    branch_index: int,
    branch_name: str,
) -> Solution:
    """
    Copy a solution and assign a globally unique identifier while retaining
    the originating exclusion branch.
    """
    details = dict(
        getattr(solution, "evaluation_details", {}) or {}
    )
    details["subproblem_index"] = branch_index
    details["subproblem_branch"] = branch_name
    details["original_solution_id"] = str(solution.id)

    return Solution(
        id=(
            f"SP{branch_index + 1}::"
            f"{branch_name}::"
            f"{solution.id}"
        ),
        selected_ids=set(solution.selected_ids),
        objectives=dict(solution.objectives),
        attributes=dict(
            getattr(solution, "attributes", {}) or {}
        ),
        constraints=dict(
            getattr(solution, "constraints", {}) or {}
        ),
        is_feasible=solution.is_feasible,
        evaluation_details=details,
        rank=getattr(solution, "rank", None),
        crowding_distance=getattr(
            solution,
            "crowding_distance",
            None,
        ),
    )


def combine_partial_pareto_fronts(
    partial_fronts: Sequence[ParetoFront],
    branch_names: Sequence[str],
    objective_directions: Dict[str, str],
) -> ParetoFront:
    """
    Combine partial Pareto fronts and retain the globally non-dominated
    feasible release compositions.
    """
    combined_solutions: List[Solution] = []
    seen_compositions = set()

    for branch_index, partial_front in enumerate(
        partial_fronts
    ):
        branch_name = (
            branch_names[branch_index]
            if branch_index < len(branch_names)
            else f"Subproblem {branch_index + 1}"
        )

        for solution in partial_front.solutions:
            if not solution.is_feasible:
                continue

            composition_key = (
                frozenset(
                    str(item_id)
                    for item_id in solution.selected_ids
                ),
                tuple(
                    sorted(
                        (
                            str(name),
                            float(value),
                        )
                        for name, value
                        in solution.objectives.items()
                    )
                ),
            )

            # The same release may appear through more than one branch.
            # It should occur once in the global front.
            if composition_key in seen_compositions:
                continue

            seen_compositions.add(composition_key)

            combined_solutions.append(
                _copy_solution_with_branch_provenance(
                    solution=solution,
                    branch_index=branch_index,
                    branch_name=branch_name,
                )
            )

    combined_front = ParetoFront(combined_solutions)

    global_front = combined_front.filter_non_dominated(
        objective_directions
    )
    global_front.set_objective_directions(
        objective_directions
    )

    return global_front

def run_optimization_models(
    models: Sequence[OptimizationModel],
    objectives: Dict[str, str],
    constraints: Dict[str, object],
    solver_factory: Callable[[], BaseSolver],
    primary_constraint_attr: str | None = None,
) -> CompositeOptimizationRunResult:
    """
    Formulate and solve one OptimizationProblem per processed model variant,
    then combine their partial Pareto fronts by dominance.

    A new solver instance is created per subproblem to avoid leaking mutable
    state between solver executions.
    """
    if not models:
        raise ValueError(
            "At least one OptimizationModel is required."
        )

    partial_results: List[OptimizationRunResult] = []
    partial_fronts: List[ParetoFront] = []
    branch_names: List[str] = []

    started_at = time.perf_counter()

    for branch_index, model in enumerate(models):
        problem = OptimizationProblem(
            model=model,
            objectives=dict(objectives),
            constraints=dict(constraints),
            primary_constraint_attr=primary_constraint_attr,
        )

        solver = solver_factory()

        result = run_optimization(
            problem=problem,
            solver=solver,
        )

        branch_name = getattr(
            model,
            "branch_name",
            f"Subproblem {branch_index + 1}",
        )

        if len(result.pareto_front) > 0:
            result.pareto_front.set_objective_directions(
                problem.get_objective_directions()
            )        

        partial_results.append(result)
        partial_fronts.append(result.pareto_front)
        branch_names.append(branch_name)

    objective_directions = {
        str(name): str(direction).lower()
        for name, direction in objectives.items()
    }

    global_front = combine_partial_pareto_fronts(
        partial_fronts=partial_fronts,
        branch_names=branch_names,
        objective_directions=objective_directions,
    )

    total_execution_time = (
        time.perf_counter() - started_at
    )

    return CompositeOptimizationRunResult(
        partial_results=partial_results,
        pareto_front=global_front,
        pareto_df=global_front.to_dataframe(),
        execution_time=total_execution_time,
    )

