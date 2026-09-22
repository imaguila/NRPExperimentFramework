"""
Application-layer data transfer objects.

These objects describe inputs and outputs exchanged between presentation
clients and application use cases. They do not contain Streamlit state or
presentation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.domain.core.model import OptimizationModel

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.domain.core.paretofront import ParetoFront
    from src.solving.core.problem import OptimizationProblem
    from src.solving.core.solver import BaseSolver



@dataclass(frozen=True)
class JsonCaseInspection:
    """
    Result of inspecting a JSON optimisation instance.

    Inspection loads the instance with its declared default aggregation
    rules so presentation clients can discover available attributes and
    relationships before requesting the final preprocessing operation.
    """

    raw_data: Dict[str, Any]
    inspected_model: OptimizationModel


@dataclass(frozen=True)
class LoadJsonCaseResult:
    """
    Result of loading and preprocessing a JSON optimisation instance.

    Attributes:
        raw_data:
            Original parsed JSON dictionary.

        base_model:
            Model produced by the loader after multivalued resolution and
            before coupling, value-dependency processing, and exclusions.

        processed_models:
            Model variants returned by the preprocessing pipeline.
    """

    raw_data: Dict[str, Any]
    base_model: OptimizationModel
    processed_models: List[OptimizationModel]



@dataclass
class OptimizationRunResult:
    """
    Result produced by one optimisation execution.

    This DTO contains the objects required by interactive and headless
    clients without depending on Streamlit.
    """

    problem: "OptimizationProblem"
    solver: "BaseSolver"
    pareto_front: "ParetoFront"
    pareto_df: pd.DataFrame
    execution_time: float

