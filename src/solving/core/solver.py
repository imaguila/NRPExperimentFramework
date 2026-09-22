# Location: src/solving/core/solver.py
"""
Abstract base interface for optimisation solvers.

This module defines the BaseSolver abstract class, which serves as a
contract for all optimisation algorithms in the framework. Solvers are
domain‑agnostic and operate on a generic OptimizationProblem, returning
a ParetoFront.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from src.domain.core.paretofront import ParetoFront
from src.solving.core.parameter import ParameterSpec
from src.solving.core.problem import OptimizationProblem


class BaseSolver(ABC):
    """
    Strategy interface for optimisation algorithms.

    This abstract class defines the minimal contract that any solver must
    implement. It is intentionally free of domain‑specific knowledge:
        - no NRP rules
        - no UI logic
        - no business assumptions
        - only the generic optimisation contract

    Subclasses must implement `solve()` to perform the actual optimisation,
    and `get_parameter_schema()` to describe their configurable parameters.
    """

    @classmethod
    @abstractmethod
    def get_parameter_schema(cls) -> List[ParameterSpec]:
        """
        Return the list of configurable parameters for this solver.

        The schema is used by the UI to generate dynamic controls
        (sliders, dropdowns, etc.) for tuning the solver.

        Returns:
            A list of ParameterSpec objects describing each parameter.
        """
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        """
        Return the human‑readable name of the solver class.

        By default, this is the class name. Subclasses may override this
        if a different display name is desired.

        Returns:
            The name of the solver class.
        """
        return cls.__name__

    @abstractmethod
    def solve(self, problem: OptimizationProblem, **kwargs: Any) -> ParetoFront:
        """
        Solve a generic optimisation problem and return a Pareto front.

        The solver uses the provided problem definition (model, objectives,
        constraints) and any additional keyword arguments (which should
        match the parameters defined in the schema).

        Args:
            problem: The optimisation problem to solve.
            **kwargs: Solver‑specific parameters (e.g., population size,
                number of generations, etc.).

        Returns:
            A ParetoFront containing the non‑dominated solutions found.
        """
        raise NotImplementedError