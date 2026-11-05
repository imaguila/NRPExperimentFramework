# Location: src/solving/core/__init__.py
"""
Core solving package.
"""

from src.solving.core.problem import OptimizationProblem, ConstraintSpec, ThresholdOperator
from src.solving.core.solver import BaseSolver
from src.solving.core.evaluator import SolutionEvaluator
from src.solving.core.registry import register_solver, get_registered_solvers, get_solver_class
from src.solving.core.parameter import ParameterSpec

__all__ = [
    "OptimizationProblem",
    "ConstraintSpec",
    "ThresholdOperator",
    "BaseSolver",
    "SolutionEvaluator",
    "register_solver",
    "get_registered_solvers",
    "get_solver_class",
    "ParameterSpec",
]