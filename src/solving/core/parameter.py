# Location: src/solving/core/parameter.py
"""
Configurable parameter specification for solvers.

This module defines the ParameterSpec dataclass, which provides an
agnostic way to describe solver hyperparameters. The UI can use this
specification to generate dynamic controls (sliders, inputs, dropdowns).
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Union, Type, Dict


@dataclass
class ParameterSpec:
    """
    Agnostic specification of a solver configuration parameter.

    This allows the UI to generate dynamic controls (sliders, inputs, dropdowns)
    based on the parameter's type, range, default value, and other constraints.

    Attributes:
        name: Internal name of the parameter (used in __init__).
        display_name: Human‑readable label for the UI.
        param_type: Python type of the parameter (int, float, bool, str).
        default: Default value.
        min_value: Minimum allowed value (for numeric types).
        max_value: Maximum allowed value (for numeric types).
        step: Step size (for numeric sliders).
        options: List of options (for dropdown selection).
        description: Help text for tooltips.
    """
    name: str
    display_name: str
    param_type: Type
    default: Any
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    options: Optional[List[Any]] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the parameter specification to a dictionary for the UI.

        The dictionary includes the parameter's name, display name, type,
        default, min/max, step, options, and description. The type is
        converted to a string representation.

        Returns:
            A dictionary containing all parameter metadata.
        """
        return {
            "name": self.name,
            "display_name": self.display_name,
            "type": self.param_type.__name__ if hasattr(self.param_type, "__name__") else str(self.param_type),
            "default": self.default,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "options": self.options,
            "description": self.description
        }