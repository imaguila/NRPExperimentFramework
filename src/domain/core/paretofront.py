# Location: src/domain/core/paretofront.py
"""
Domain-agnostic representation of a Pareto Front collection.
"""

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import copy
import pandas as pd

from src.domain.core.solution import Solution


logger = logging.getLogger(__name__)


class ParetoFront:
    """
    Domain-agnostic collection of solutions forming a Pareto Front.

    Provides set operations such as filtering, sorting, and conversion
    to DataFrames, as well as import/export to CSV and JSON formats.
    """

    def __init__(self, solutions: Optional[List[Solution]] = None):
        self.solutions: List[Solution] = solutions or []

    def __len__(self) -> int:
        return len(self.solutions)

    def __iter__(self) -> Iterator[Solution]:
        return iter(self.solutions)

    def __getitem__(self, index: int) -> Solution:
        return self.solutions[index]

    def __repr__(self) -> str:
        feasible_count = sum(1 for s in self.solutions if s.is_feasible)
        return f"<ParetoFront(size={len(self.solutions)}, feasible={feasible_count})>"

    def add_solution(self, solution: Solution) -> None:
        """Append a single solution to the front."""
        self.solutions.append(solution)

    def get_by_id(self, sol_id: str) -> Optional[Solution]:
        """Retrieve a solution by its unique identifier."""
        for s in self.solutions:
            if str(s.id) == str(sol_id):
                return s
        return None

    def filter_feasible(self) -> "ParetoFront":
        """Return a new ParetoFront containing only feasible solutions."""
        return ParetoFront([s for s in self.solutions if s.is_feasible])

    def sort_by_objective(self, objective_name: str, ascending: bool = True) -> List[Solution]:
        """Return the list of solutions sorted by a specific objective value."""
        return sorted(
            self.solutions,
            key=lambda s: s.objectives.get(objective_name, 0.0),
            reverse=not ascending
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Export the entire Pareto Front to a Pandas DataFrame."""
        if not self.solutions:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self.solutions])

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        decision_cols: Optional[List[str]] = None,
        objective_cols: Optional[List[str]] = None,
        constraint_cols: Optional[List[str]] = None,
        objective_directions: Optional[Dict[str, str]] = None,
    ) -> "ParetoFront":
        """
        Build a ParetoFront from a DataFrame (loaded from CSV or memory).

        Args:
            df: Input DataFrame.
            decision_cols: Column names representing decision variables.
            objective_cols: Column names representing objective values.
            constraint_cols: Column names representing constraints.
            objective_directions: Mapping of objective names to 'max' or 'min'.

        Returns:
            A ParetoFront instance.
        """
        if df is None or df.empty:
            return cls([])

        if decision_cols is None:
            decision_cols = [col for col in df.columns if col.startswith("req_")]

        solutions = [
            Solution.from_dict(
                data=row.to_dict(),
                decision_cols=decision_cols,
                objective_cols=objective_cols,
                constraint_cols=constraint_cols,
                sol_id=row.get("id", f"sol_{idx}")
            )
            for idx, row in df.iterrows()
        ]
        front = cls(solutions)
        if objective_directions:
            front.set_objective_directions({str(k): str(v).lower() for k, v in objective_directions.items()})
        return front

    def filter_non_dominated(self, objective_directions: Optional[Dict[str, str]] = None) -> "ParetoFront":
        """
        Return a new ParetoFront containing only non‑dominated solutions.

        Args:
            objective_directions: Objective directions (max/min) for dominance checks.

        Returns:
            A ParetoFront with only the non‑dominated solutions.
        """
        non_dominated = []
        for i, s1 in enumerate(self.solutions):
            dominated = False
            for j, s2 in enumerate(self.solutions):
                if i != j and s2.dominates(s1, objective_directions):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(s1)
        return ParetoFront(non_dominated)

    # =========================================================================
    # EXPORT TO DICTIONARY / JSON
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Pareto Front to a dictionary structure."""
        serialized_solutions = []
        for idx, sol in enumerate(self.solutions, start=1):
            evals = getattr(sol, "objectives", getattr(sol, "evaluations", {}))
            selected = sorted(list(getattr(sol, "selected_ids", [])))
            sol_dict = {
                "solution_id": idx,
                "selected_items": selected,
                "objectives": evals,
                "attributes": dict(getattr(sol, "attributes", {})),
            }
            serialized_solutions.append(sol_dict)

        payload = {
            "total_solutions": len(self.solutions),
            "solutions": serialized_solutions,
        }
        if hasattr(self, "objective_directions"):
            payload["objective_directions"] = dict(getattr(self, "objective_directions"))
        return payload

    def export_to_json(self, filepath: Union[str, Path]) -> None:
        """Export the Pareto Front to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    # =========================================================================
    # EXPORT TO CSV (BINARY MATRIX AND OBJECTIVES)
    # =========================================================================

    def to_csv(self, all_item_ids: Optional[List[str]] = None) -> str:
        """
        Generate a CSV string representing the binary decision matrix
        along with objective values for each solution.

        Args:
            all_item_ids: Optional ordered list of all possible item IDs.
                If not provided, the union of selected items across solutions is used.

        Returns:
            CSV string with header: objectives columns + item columns.
        """
        if not self.solutions:
            return ""

        # 1. Collect objective names (preserve order of appearance)
        objective_names = []
        for sol in self.solutions:
            evals = getattr(sol, "objectives", getattr(sol, "evaluations", {}))
            for k in evals.keys():
                if k not in objective_names:
                    objective_names.append(k)

        # 2. Determine the universe of item/requirement columns
        if all_item_ids is not None:
            item_columns = list(all_item_ids)
        else:
            all_selected = set()
            for sol in self.solutions:
                selected = getattr(sol, "selected_ids", set())
                all_selected.update(selected)
            item_columns = sorted(list(all_selected))

        # 3. Build header: objectives + items
        header = objective_names + item_columns

        # 4. Generate rows
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(header)

        for sol in self.solutions:
            evals = getattr(sol, "objectives", getattr(sol, "evaluations", {}))
            selected_set = set(getattr(sol, "selected_ids", []))

            row_objectives = [evals.get(obj_name, 0.0) for obj_name in objective_names]
            row_items = [1 if item_id in selected_set else 0 for item_id in item_columns]

            writer.writerow(row_objectives + row_items)

        return output.getvalue()

    def export_to_csv(
        self,
        filepath: Union[str, Path],
        all_item_ids: Optional[List[str]] = None
    ) -> None:
        """
        Save the Pareto Front to a CSV file.

        Args:
            filepath: Destination file path.
            all_item_ids: Optional list of all item IDs in the problem.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        csv_content = self.to_csv(all_item_ids=all_item_ids)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(csv_content)

    def clone(self) -> "ParetoFront":
        """Create an independent deep copy of the Pareto Front."""
        return copy.deepcopy(self)

    # =========================================================================
    # INTEGRATION WITH SOLUTION SET COLLECTION
    # =========================================================================

    def to_solution_set(
        self,
        name: str = "Pareto Front",
        category: str = "pareto_front",
        algorithm: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SolutionSet":
        """
        Convert this Pareto Front into a SolutionSet with metadata.

        Returns:
            A SolutionSet instance.
        """
        from src.domain.core.collection import SolutionSet
        return SolutionSet(
            name=name,
            front=self,
            category=category,
            algorithm=algorithm,
            metadata=metadata or {},
        )

    def add_to_collection(
        self,
        collection: Any,
        name: str = "Pareto Front",
        category: str = "pareto_front",
        algorithm: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SolutionSet":
        """
        Register this Pareto Front directly into a SolutionSetCollection.

        Returns:
            The created SolutionSet.
        """
        s_set = self.to_solution_set(
            name=name,
            category=category,
            algorithm=algorithm,
            metadata=metadata,
        )
        collection.add_set(s_set)
        return s_set

    # =========================================================================
    # LOAD FROM CSV
    # =========================================================================

    @classmethod
    def from_csv(
        cls,
        filepath_or_buffer: Union[str, Path, io.StringIO],
        objective_cols: Optional[List[str]] = None,
        decision_prefix: str = "req_",
        id_col: Optional[str] = None,
        objective_directions: Optional[Dict[str, str]] = None,
    ) -> "ParetoFront":
        """
        Load a Pareto Front from a CSV file.

        Args:
            filepath_or_buffer: Path to CSV file or a StringIO buffer.
            objective_cols: List of objective column names.
                If None, all numeric columns not starting with decision_prefix are used.
            decision_prefix: Prefix for decision variable columns (e.g., 'req_').
            id_col: Name of the column containing solution IDs.
                If None, auto-generated IDs (sol_0, sol_1, ...) are used.
            objective_directions: Mapping of objective names to 'max' or 'min'.

        Returns:
            A reconstructed ParetoFront instance.
        """
        if isinstance(filepath_or_buffer, (str, Path)):
            df = pd.read_csv(filepath_or_buffer)
        else:
            df = pd.read_csv(filepath_or_buffer)

        decision_cols = [col for col in df.columns if col.startswith(decision_prefix)]

        if objective_cols is None:
            objective_cols = [
                col for col in df.columns
                if col not in decision_cols
                and pd.api.types.is_numeric_dtype(df[col])
            ]

        if id_col and id_col in df.columns:
            df["id"] = df[id_col].astype(str)

        front = cls.from_dataframe(
            df=df,
            decision_cols=decision_cols,
            objective_cols=objective_cols,
            objective_directions=objective_directions,
        )
        if objective_directions:
            front.set_objective_directions({str(k): str(v).lower() for k, v in objective_directions.items()})
        return front

    # -------------------------------------------------------------------------
    # Additional serialization methods (duplicated to_dict is kept for compatibility)
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Pareto Front to a dictionary (serializable)."""
        serialized_solutions = []
        for idx, sol in enumerate(self.solutions, start=1):
            evals = getattr(sol, "objectives", getattr(sol, "evaluations", {}))
            selected = sorted(list(getattr(sol, "selected_ids", [])))

            sol_dict = {
                "solution_id": idx,
                "selected_items": selected,
                "objectives": evals
            }
            serialized_solutions.append(sol_dict)

        payload = {
            "total_solutions": len(self.solutions),
            "solutions": serialized_solutions,
        }
        if hasattr(self, "objective_directions"):
            payload["objective_directions"] = dict(getattr(self, "objective_directions"))
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParetoFront":
        """
        Reconstruct a ParetoFront from a dictionary (as produced by to_dict).

        Args:
            data: Dictionary containing the serialized front.

        Returns:
            A ParetoFront instance.
        """
        solutions = []
        for sol_data in data.get("solutions", []):
            selected_ids = set(sol_data.get("selected_items", []))
            objectives = sol_data.get("objectives", {})
            attributes = sol_data.get("attributes", {}) or {}
            sol = Solution(
                id=f"sol_{sol_data.get('solution_id', 0)}",
                selected_ids=selected_ids,
                objectives=objectives,
                attributes={str(k): float(v) for k, v in attributes.items() if isinstance(v, (int, float))},
                is_feasible=True
            )
            solutions.append(sol)
        front = cls(solutions)
        directions = data.get("objective_directions")
        if directions:
            front.set_objective_directions({str(k): str(v).lower() for k, v in directions.items()})
        return front

    # -------------------------------------------------------------------------
    # Objective directions storage
    # -------------------------------------------------------------------------

    def set_objective_directions(self, directions: Dict[str, str]) -> None:
        """
        Store the objective directions (max/min) in the front.

        Args:
            directions: Mapping of objective names to 'max' or 'min'.
        """
        self.objective_directions = directions

    # -------------------------------------------------------------------------
    # Quality metrics computation
    # -------------------------------------------------------------------------

    def compute_metrics(
        self,
        reference_point: Optional[List[float]] = None,
        objective_names: Optional[List[str]] = None,
        directions: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Compute quality metrics for the Pareto front.

        Args:
            reference_point: Reference point for Hypervolume.
            objective_names: Objectives to consider (all if None).
            directions: Objective directions (max/min). If None, stored directions are used.

        Returns:
            Dictionary with metric names and values.
        """
        from src.solving.core.metrics import ParetoMetrics

        # Use stored directions if not explicitly provided
        if directions is None and hasattr(self, "objective_directions"):
            directions = self.objective_directions

        return ParetoMetrics.compute_all_metrics(
            front=self,
            reference_point=reference_point,
            objective_names=objective_names,
            directions=directions
        )


# -------------------------------------------------------------------------
# Wrapper function for convenience
# -------------------------------------------------------------------------

def load_pareto_csv(
    filepath_or_buffer: Union[str, Path, io.StringIO],
    objective_columns: Optional[List[str]] = None,
    decision_prefix: str = "req_",
    objective_directions: Optional[dict] = None,
) -> Tuple[ParetoFront, pd.DataFrame]:
    """
    Convenience wrapper to load a Pareto Front from CSV.

    Args:
        filepath_or_buffer: Path or buffer.
        objective_columns: List of objective column names.
        decision_prefix: Prefix for decision variables.
        objective_directions: Mapping of objective names to 'max' or 'min'.

    Returns:
        A tuple (ParetoFront, DataFrame) containing the front and its DataFrame.
    """
    front = ParetoFront.from_csv(
        filepath_or_buffer=filepath_or_buffer,
        objective_cols=objective_columns,
        decision_prefix=decision_prefix,
        objective_directions=objective_directions,
    )
    return front, front.to_dataframe()