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

from src.application.dto import OptimizationRunResult
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