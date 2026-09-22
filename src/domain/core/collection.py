# Location: src/domain/core/collection.py
"""
Solution Set Collection Core Representation.

This module provides generic containers for managing multiple Pareto fronts,
Sets of Interest (SOIs), clusters, or filtered subsets. The components are
completely domain‑agnostic and designed for flexible storage, retrieval,
and serialisation of solution sets.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional, Union
import uuid

import pandas as pd

from src.domain.core.paretofront import ParetoFront
from src.domain.core.solution import Solution


@dataclass
class SolutionSet:
    """
    A named collection of solutions with provenance metadata.

    Encapsulates a ParetoFront (or any subset of solutions) and adds
    identity, category, algorithm name, and custom metadata. This allows
    distinguishing between different runs, algorithms, or manually curated
    subsets (SOIs).

    Attributes:
        id: Unique identifier (auto‑generated UUID if not provided).
        name: Human‑readable name for the set.
        front: The ParetoFront instance containing the solutions.
        category: Classification of the set (e.g., 'pareto_front', 'soi',
            'cluster', 'filtered').
        algorithm: Name of the algorithm that generated the set (if applicable).
        metadata: Arbitrary key‑value metadata for provenance or UI purposes.
        created_at: Unix timestamp of creation (auto‑set to current time).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Unnamed Set"
    front: ParetoFront = field(default_factory=ParetoFront)
    category: str = "pareto_front"  # 'pareto_front', 'soi', 'cluster', 'filtered'
    algorithm: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __len__(self) -> int:
        """Return the number of solutions in the set."""
        return len(self.front)

    def __iter__(self) -> Iterator[Solution]:
        """Iterate over the solutions in the set."""
        return iter(self.front)

    @property
    def solutions(self) -> List[Solution]:
        """Return the list of solutions in the set."""
        return self.front.solutions

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the SolutionSet to a serialisable dictionary.

        Returns:
            Dictionary containing all metadata and the serialised front.
        """
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "algorithm": self.algorithm,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "front": self.front.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SolutionSet":
        """
        Reconstruct a SolutionSet from a dictionary.

        Args:
            data: Dictionary containing serialised set data.

        Returns:
            A SolutionSet instance.
        """
        raw_front = data.get("front", {})
        solutions_raw = raw_front.get("solutions", []) if isinstance(raw_front, dict) else []

        # Reconstruct each Solution from its dictionary representation
        solutions = [
            Solution.from_dict(data=s) if isinstance(s, dict) else s
            for s in solutions_raw
        ]

        return cls(
            id=str(data.get("id", uuid.uuid4())),
            name=str(data.get("name", "Unnamed Set")),
            front=ParetoFront(solutions),
            category=str(data.get("category", "pareto_front")),
            algorithm=data.get("algorithm"),
            metadata=data.get("metadata", {}),
            created_at=float(data.get("created_at", time.time())),
        )


class SolutionSetCollection:
    """
    Global collection of multiple SolutionSets.

    This container holds several sets (Pareto fronts from different runs,
    algorithms, seeds, or user‑saved SOIs) and provides operations for
    adding, retrieving, filtering, and exporting them. It also supports
    merging all solutions into a global Pareto front.

    Attributes:
        id: Unique identifier for the collection.
        name: Human‑readable name for the collection.
        sets: Dictionary mapping set IDs to SolutionSet instances.
    """

    def __init__(self, name: str = "Optimization Study", sets: Optional[List[SolutionSet]] = None):
        """
        Initialise a new SolutionSetCollection.

        Args:
            name: Display name for the collection.
            sets: Optional initial list of SolutionSet objects.
        """
        self.id: str = str(uuid.uuid4())
        self.name: str = name
        self.sets: Dict[str, SolutionSet] = {}

        if sets:
            for s in sets:
                self.add_set(s)

    def __len__(self) -> int:
        """Return the number of stored SolutionSets."""
        return len(self.sets)

    def __iter__(self) -> Iterator[SolutionSet]:
        """Iterate over the stored SolutionSets."""
        return iter(self.sets.values())

    def __getitem__(self, key: str) -> SolutionSet:
        """Retrieve a SolutionSet by its ID."""
        return self.sets[key]

    def add_set(self, solution_set: SolutionSet) -> None:
        """
        Add a SolutionSet to the collection.

        Args:
            solution_set: The SolutionSet instance to add.
        """
        self.sets[solution_set.id] = solution_set

    def create_and_add_set(
        self,
        name: str,
        pareto_front: ParetoFront,
        category: str = "pareto_front",
        algorithm: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SolutionSet:
        """
        Create a new SolutionSet from a ParetoFront and add it to the collection.

        Args:
            name: Name for the new set.
            pareto_front: ParetoFront instance to wrap.
            category: Category for the set.
            algorithm: Algorithm name (optional).
            metadata: Additional metadata (optional).

        Returns:
            The newly created SolutionSet.
        """
        s_set = SolutionSet(
            name=name,
            front=pareto_front,
            category=category,
            algorithm=algorithm,
            metadata=metadata or {},
        )
        self.add_set(s_set)
        return s_set

    def get_by_name(self, name: str) -> Optional[SolutionSet]:
        """
        Retrieve a SolutionSet by its exact name.

        Args:
            name: Name of the set.

        Returns:
            The SolutionSet if found, otherwise None.
        """
        for s in self.sets.values():
            if s.name == name:
                return s
        return None

    def get_by_category(self, category: str) -> List[SolutionSet]:
        """
        Retrieve all SolutionSets belonging to a given category.

        Args:
            category: Category name (e.g., 'pareto_front', 'soi').

        Returns:
            List of matching SolutionSet objects.
        """
        return [s for s in self.sets.values() if s.category == category]

    def remove_set(self, set_id_or_name: str) -> bool:
        """
        Remove a SolutionSet by its ID or name.

        Args:
            set_id_or_name: Either the set ID or the set name.

        Returns:
            True if a set was removed, False otherwise.
        """
        if set_id_or_name in self.sets:
            del self.sets[set_id_or_name]
            return True
        for sid, s in list(self.sets.items()):
            if s.name == set_id_or_name:
                del self.sets[sid]
                return True
        return False

    def get_all_unique_solutions(self) -> Dict[str, Solution]:
        """
        Extract the union of all unique solutions across all stored sets.

        Returns:
            Dictionary mapping solution IDs to Solution objects.
        """
        unique_solutions: Dict[str, Solution] = {}
        for s_set in self.sets.values():
            for sol in s_set.front:
                if sol.id not in unique_solutions:
                    unique_solutions[sol.id] = sol
        return unique_solutions

    def compute_global_pareto_front(
        self, objective_directions: Optional[Dict[str, str]] = None
    ) -> ParetoFront:
        """
        Combine all solutions from all stored sets and compute the global
        non‑dominated Pareto front.

        Args:
            objective_directions: Mapping of objective names to 'max' or 'min'
                for the dominance check.

        Returns:
            A new ParetoFront containing only the non‑dominated solutions
            from the combined set.
        """
        all_unique = list(self.get_all_unique_solutions().values())
        combined_front = ParetoFront(all_unique)
        return combined_front.filter_non_dominated(objective_directions)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all SolutionSets to a consolidated Pandas DataFrame.

        Each row corresponds to a single solution, with additional columns
        indicating the set ID, name, category, and algorithm.

        Returns:
            DataFrame containing all solutions with provenance metadata.
        """
        rows = []
        for s_set in self.sets.values():
            for sol in s_set.front:
                row = sol.to_dict()
                row["set_id"] = s_set.id
                row["set_name"] = s_set.name
                row["set_category"] = s_set.category
                row["set_algorithm"] = s_set.algorithm or ""
                rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entire collection to a dictionary for JSON serialisation.

        Returns:
            Dictionary with collection metadata and all stored sets.
        """
        return {
            "id": self.id,
            "name": self.name,
            "total_sets": len(self.sets),
            "sets": {sid: s.to_dict() for sid, s in self.sets.items()},
        }

    def export_to_json(self, filepath: Union[str, Path]) -> None:
        """
        Save the complete collection to a JSON file.

        Args:
            filepath: Destination file path. Parent directories will be
                created if they do not exist.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)