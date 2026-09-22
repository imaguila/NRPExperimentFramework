# Location: src/domain/core/solution.py
"""
Domain-agnostic dataclass representing a candidate solution in optimization models.
"""

from dataclasses import dataclass, field
from typing import Set, Dict, Any, List, Optional
import pandas as pd


@dataclass
class Solution:
    """
    Domain-agnostic representation of a candidate solution.

    Attributes:
        id: Unique identifier for the solution.
        selected_ids: Set of item identifiers selected in this solution.
        objectives: Dictionary mapping objective names to their evaluated values.
        attributes: Dictionary of additional numeric attributes (non-objective).
        constraints: Dictionary mapping constraint names to their evaluated values.
        is_feasible: Boolean indicating whether the solution satisfies all constraints.
        evaluation_details: Additional metadata or debug info from evaluation.
        rank: Rank assigned by multi-objective sorting (e.g., NSGA-II front rank).
        crowding_distance: Crowding distance for diversity maintenance (e.g., NSGA-II).
    """
    id: str
    selected_ids: Set[str]
    objectives: Dict[str, float] = field(default_factory=dict)
    attributes: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, float] = field(default_factory=dict)
    is_feasible: bool = True
    evaluation_details: Dict[str, Any] = field(default_factory=dict)
    rank: Optional[int] = None
    crowding_distance: Optional[float] = None

    @property
    def scope(self) -> int:
        """Returns the number of items selected in the solution."""
        return len(self.selected_ids)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the solution to a flat dictionary.

        This representation is suitable for DataFrame construction,
        CSV export, and UI visualization.
        Note: 'scope' is intentionally excluded here; it is computed
        on-demand by the enrichment module.
        """
        sorted_ids = sorted(list(self.selected_ids))
        res = {
            "id": self.id,
            "selected_ids": sorted_ids,
            "selected_ids_str": ", ".join(sorted_ids),
            "is_feasible": self.is_feasible,
        }
        res.update(self.objectives)
        if self.attributes:
            res["attributes"] = dict(self.attributes)
        res.update(self.constraints)
        return res

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        decision_cols: Optional[List[str]] = None,
        objective_cols: Optional[List[str]] = None,
        constraint_cols: Optional[List[str]] = None,
        sol_id: Optional[str] = None
    ) -> "Solution":
        """
        Reconstructs a Solution object from a flat dictionary or DataFrame row.

        Args:
            data: Dictionary containing the solution data.
            decision_cols: List of column names representing decision variables.
            objective_cols: List of column names representing objective values.
            constraint_cols: List of column names representing constraint values.
            sol_id: Optional override for the solution ID.

        Returns:
            A Solution instance.
        """
        actual_id = str(sol_id or data.get("id", "sol_unknown"))

        # 1. Extract selected_ids
        selected_ids: Set[str] = set()
        if "selected_ids" in data and data["selected_ids"]:
            val = data["selected_ids"]
            if isinstance(val, (list, tuple, set)):
                selected_ids = {str(item) for item in val}
            elif isinstance(val, str):
                selected_ids = {s.strip() for s in val.split(",") if s.strip()}
        elif "selected_ids_str" in data and isinstance(data["selected_ids_str"], str):
            selected_ids = {s.strip() for s in data["selected_ids_str"].split(",") if s.strip()}
        elif decision_cols:
            for col in decision_cols:
                if col in data:
                    v = data[col]
                    if v == 1 or v is True or str(v).lower() in ("true", "1", "yes"):
                        selected_ids.add(str(col))

        # 2. Extract Objectives
        objectives: Dict[str, float] = {}
        if objective_cols:
            for col in objective_cols:
                if col in data and data[col] is not None:
                    try:
                        objectives[col] = float(data[col])
                    except (ValueError, TypeError):
                        pass
        elif "objectives" in data and isinstance(data["objectives"], dict):
            objectives = {k: float(v) for k, v in data["objectives"].items()}

        # 3. Extract Constraints
        constraints: Dict[str, float] = {}
        if constraint_cols:
            for col in constraint_cols:
                if col in data and data[col] is not None:
                    try:
                        constraints[col] = float(data[col])
                    except (ValueError, TypeError):
                        pass
        elif "constraints" in data and isinstance(data["constraints"], dict):
            constraints = {k: float(v) for k, v in data["constraints"].items()}

        # 4. Extract additional numeric attributes not covered by objectives or constraints
        attributes: Dict[str, float] = {}
        if "attributes" in data and isinstance(data["attributes"], dict):
            attributes = {str(k): float(v) for k, v in data["attributes"].items() if isinstance(v, (int, float))}

        if not attributes:
            known_keys = set()
            if objective_cols:
                known_keys.update(objective_cols)
            if decision_cols:
                known_keys.update(decision_cols)
            if constraint_cols:
                known_keys.update(constraint_cols)
            known_keys.update({"id", "selected_ids", "selected_ids_str", "is_feasible", "constraints", "attributes"})
            for key, value in data.items():
                if key in known_keys:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    try:
                        attributes[str(key)] = float(value)
                    except (TypeError, ValueError):
                        pass

        # 5. Feasibility and metadata
        is_feasible = bool(data.get("is_feasible", True))
        rank = data.get("rank")
        crowding = data.get("crowding_distance")

        return cls(
            id=actual_id,
            selected_ids=selected_ids,
            objectives=objectives,
            attributes=attributes,
            constraints=constraints,
            is_feasible=is_feasible,
            rank=int(rank) if rank is not None else None,
            crowding_distance=float(crowding) if crowding is not None else None,
        )

    @staticmethod
    def to_dataframe(solutions: List["Solution"]) -> pd.DataFrame:
        """
        Converts a list of Solution objects into a Pandas DataFrame.

        Args:
            solutions: List of Solution instances.

        Returns:
            DataFrame containing the flat representation of each solution.
        """
        if not solutions:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in solutions])

    def dominates(
        self,
        other: "Solution",
        objective_directions: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Determines whether this solution dominates another.

        Dominance is evaluated according to the specified objective directions
        (maximize or minimize), applying a feasibility rule:
            - A feasible solution always dominates an infeasible one.
            - If both are infeasible, neither dominates the other (no objective comparison).

        Args:
            other: The solution to compare against.
            objective_directions: Mapping of objective names to 'max' or 'min'.
                                   If None, all objectives are assumed to be maximized.

        Returns:
            True if this solution dominates the other, False otherwise.
        """
        # 1. Feasibility rule (Constrained Dominance)
        if self.is_feasible and not other.is_feasible:
            return True
        if not self.is_feasible and other.is_feasible:
            return False
        if not self.is_feasible and not other.is_feasible:
            return False

        # 2. Pareto dominance for feasible solutions
        if objective_directions is None:
            # Default: maximize all objectives
            objective_directions = {k: "max" for k in self.objectives.keys()}

        better_in_at_least_one = False
        for obj_name, direction in objective_directions.items():
            v1 = self.objectives.get(obj_name, 0.0)
            v2 = other.objectives.get(obj_name, 0.0)
            is_max = str(direction).lower().startswith("max")

            if is_max:
                if v1 < v2:
                    return False
                if v1 > v2:
                    better_in_at_least_one = True
            else:  # minimization
                if v1 > v2:
                    return False
                if v1 < v2:
                    better_in_at_least_one = True

        return better_in_at_least_one

    def __repr__(self) -> str:
        return (
            f"Solution(id='{self.id}', scope={self.scope}, "
            f"feasible={self.is_feasible}, objectives={self.objectives})"
        )