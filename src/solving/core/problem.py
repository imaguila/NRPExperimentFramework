# Location: src/solving/core/problem.py
"""
Domain‑agnostic optimisation problem specification.

This module defines the core data structures for specifying an optimisation
problem, including objectives, constraints, and the problem container itself.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from src.domain.core.collection import SolutionSet, SolutionSetCollection
from src.domain.core.model import OptimizationModel
from src.domain.core.paretofront import ParetoFront


class ThresholdOperator(str, Enum):
    """
    Operators for threshold‑based constraints.

    Defines the comparison operators supported for constraint specifications.
    """
    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="
    EQUAL = "=="
    BETWEEN = "between"


class ObjectiveSense(str, Enum):
    """
    Objective optimisation direction.

    Indicates whether an objective should be maximised or minimised.
    """
    MAXIMIZE = "max"
    MINIMIZE = "min"


@dataclass
class ConstraintSpec:
    """
    Specification of a threshold constraint on an attribute.

    Attributes:
        operator: Comparison operator (e.g., <=, >=, ==, between).
        value: Threshold value or tuple (min, max) for 'between'.
        attribute: Name of the attribute this constraint applies to.
    """
    operator: Union[ThresholdOperator, str]
    value: Union[float, Tuple[float, float]]
    attribute: Optional[str] = None


@dataclass
class OptimizationProblem:
    """
    Complete specification of an optimisation instance and its results.

    This class encapsulates the model, objective directions, constraints,
    and a collection of Pareto fronts generated from previous runs.

    Attributes:
        model: The domain‑agnostic optimisation model.
        objectives: Mapping of attribute names to 'max' or 'min' directions.
        constraints: Dictionary of constraint specifications per attribute.
        primary_constraint_attr: Primary constraint attribute (optional, for
            specific solver heuristics).
        solutions_collection: Container for storing multiple Pareto fronts/SOIs.
    """
    model: OptimizationModel
    objectives: Dict[str, Union[ObjectiveSense, str]]  # e.g., {"satisfaction": "max", "effort": "min"}
    constraints: Dict[str, ConstraintSpec] = field(default_factory=dict)
    primary_constraint_attr: Optional[str] = None
    solutions_collection: SolutionSetCollection = field(default_factory=SolutionSetCollection)

    def __post_init__(self) -> None:
        """
        Ensure the solutions collection has a meaningful name.

        If the collection has no name or a default name, it is set based on
        the model's name.
        """
        if not self.solutions_collection.name or self.solutions_collection.name == "Optimization Study":
            self.solutions_collection.name = f"Results - {self.model.name}"

    def is_maximize(self, objective_name: str) -> bool:
        """
        Check whether a given objective should be maximised.

        Args:
            objective_name: Name of the objective.

        Returns:
            True if the objective direction is 'max', False otherwise.
        """
        sense = self.objectives.get(objective_name, ObjectiveSense.MAXIMIZE)
        sense_str = sense.value if isinstance(sense, ObjectiveSense) else str(sense)
        return sense_str.lower().startswith("max")

    def get_objective_directions(self) -> Dict[str, str]:
        """
        Return a dictionary mapping each objective to 'max' or 'min'.

        This is useful for passing directly to Pareto front methods that
        require direction information.

        Returns:
            Dictionary of objective names to direction strings.
        """
        directions = {}
        for obj, sense in self.objectives.items():
            sense_str = sense.value if isinstance(sense, ObjectiveSense) else str(sense)
            directions[obj] = "max" if sense_str.lower().startswith("max") else "min"
        return directions

    def add_pareto_front(
        self,
        front: ParetoFront,
        name: str = "Pareto Front",
        category: str = "pareto_front",
        algorithm: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SolutionSet:
        """
        Explicitly register a Pareto front in the problem's collection.

        The front is wrapped in a SolutionSet, enriched with problem metadata,
        and synchronised with both the problem's and the model's collections.

        Args:
            front: The Pareto front to store.
            name: Human‑readable name for the front.
            category: Type of set (e.g., 'pareto_front', 'soi').
            algorithm: Name of the algorithm that generated the front.
            metadata: Additional metadata to attach.

        Returns:
            The created SolutionSet instance.
        """
        merged_metadata: Dict[str, Any] = dict(metadata or {})
        merged_metadata["problem"] = self.get_problem_metadata()

        s_set = front.to_solution_set(
            name=name,
            category=category,
            algorithm=algorithm,
            metadata=merged_metadata,
        )
        self.solutions_collection.add_set(s_set)

        if self.model and hasattr(self.model, "solutions_collection"):
            self.model.solutions_collection.add_set(s_set)

        return s_set

    def solve_and_store(
        self,
        solver: Any,
        name: Optional[str] = None,
        category: str = "pareto_front",
        **kwargs: Any,
    ) -> Tuple[ParetoFront, SolutionSet]:
        """
        Run a solver on this problem and store the resulting Pareto front.

        The solver is invoked with the current problem configuration, and
        the obtained front is automatically added to both the problem's
        and the model's solution collections.

        Args:
            solver: The solver instance (must implement `solve(problem, **kwargs)`).
            name: Optional name for the stored set. If None, a default is generated.
            category: Category for the stored set.
            **kwargs: Additional parameters passed to the solver.

        Returns:
            A tuple (ParetoFront, SolutionSet) containing the generated front
            and its wrapper.
        """
        front: ParetoFront = solver.solve(self, **kwargs)
        alg_name = getattr(solver, "name", solver.__class__.__name__)

        s_set = self.add_pareto_front(
            front=front,
            name=name or f"Run - {alg_name}",
            category=category,
            algorithm=alg_name,
            metadata={"parameters": kwargs},
        )
        return front, s_set

    def get_problem_metadata(self) -> Dict[str, Any]:
        """
        Serialise the complete problem configuration to a metadata dictionary.

        This is stored alongside each Pareto front to preserve the exact
        problem setup (model name, objectives, bounds, evaluation rules,
        constraints, and primary constraint attribute) that produced it.

        Returns:
            A dictionary containing the full problem configuration.
        """
        # Serialise constraints
        constraints_dict = {}
        for attr, spec in self.constraints.items():
            if hasattr(spec, "operator") and hasattr(spec, "value"):
                op_str = spec.operator.value if hasattr(spec.operator, "value") else str(spec.operator)
                constraints_dict[attr] = {"operator": op_str, "value": spec.value}
            elif isinstance(spec, dict):
                constraints_dict[attr] = spec
            else:
                constraints_dict[attr] = str(spec)

        return {
            "model_name": getattr(self.model, "name", ""),
            "objectives": self.objectives.copy(),
            "bounds": getattr(self, "bounds", {}).copy(),
            "evaluation_rules": getattr(self, "evaluation_rules", {}).copy(),
            "constraints": constraints_dict,
            "primary_constraint_attr": getattr(self, "primary_constraint_attr", None),
        }