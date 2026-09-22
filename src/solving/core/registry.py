# Location: src/solving/core/registry.py
"""
Solver registry for dynamic discovery and registration.

This module provides a global registry for optimisation solvers. Solvers are
registered using the `@register_solver` decorator and are automatically
discovered by scanning the `src.solving.algorithms` package.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Optional, Type

from src.solving.core.solver import BaseSolver

_SOLVER_REGISTRY: Dict[str, Type[BaseSolver]] = {}
_DISCOVERED = False


def register_solver(name: str):
    """
    Decorator to register a solver implementation under a user‑facing name.

    The decorator adds the decorated class to the global solver registry,
    making it available for dynamic instantiation via `get_solver_class()`.

    Args:
        name: The public name under which the solver will be registered.

    Returns:
        A decorator function that registers the solver class.
    """
    def decorator(cls: Type[BaseSolver]):
        _SOLVER_REGISTRY[name] = cls
        return cls
    return decorator


def _discover_solvers() -> None:
    """
    Discover and import all solver modules in the `src.solving.algorithms` package.

    This function is called lazily when the registry is first accessed.
    It walks through the package and imports each module, which triggers
    the execution of the `@register_solver` decorators and populates the registry.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return

    _DISCOVERED = True

    try:
        import src.solving.algorithms as algorithms_pkg
        pkg_path = algorithms_pkg.__path__
        pkg_name = algorithms_pkg.__name__

        for _, module_name, is_pkg in pkgutil.walk_packages(pkg_path, prefix=f"{pkg_name}."):
            if not is_pkg:
                importlib.import_module(module_name)
    except Exception:
        # Silently ignore import errors (e.g., missing dependencies)
        pass


def get_registered_solvers() -> Dict[str, Type[BaseSolver]]:
    """
    Retrieve all registered solvers.

    This function triggers dynamic discovery of solvers if it has not
    already been performed.

    Returns:
        A dictionary mapping solver names to their class types.
    """
    _discover_solvers()
    return dict(_SOLVER_REGISTRY)


def get_solver_class(name: str) -> Optional[Type[BaseSolver]]:
    """
    Retrieve a solver class by its registered name.

    Args:
        name: The public name of the solver.

    Returns:
        The solver class if found, otherwise None.
    """
    solvers = get_registered_solvers()
    return solvers.get(name)