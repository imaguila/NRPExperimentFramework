# Location: src/ui/utils/widget_factory.py
"""
Streamlit widget factory for ParameterSpec.

This module provides a function to render a Streamlit widget based on a
ParameterSpec, maintaining a clean separation between UI and business logic.
"""

import streamlit as st
from typing import Any

from src.solving.core.parameter import ParameterSpec


def render_parameter_widget(spec: ParameterSpec, key: str) -> Any:
    """
    Render a Streamlit widget from a ParameterSpec and return the user's value.

    The widget type is determined by the parameter's type:
        - int: number_input with integer step
        - float: number_input with float step and formatting
        - bool: checkbox
        - str with options: selectbox
        - other: text_input

    Args:
        spec: The ParameterSpec describing the parameter.
        key: Unique key for the Streamlit widget.

    Returns:
        The value selected by the user (type matches spec.param_type).
    """
    if spec.param_type == int:
        return st.number_input(
            label=spec.display_name,
            min_value=int(spec.min_value) if spec.min_value is not None else None,
            max_value=int(spec.max_value) if spec.max_value is not None else None,
            value=int(spec.default),
            step=int(spec.step) if spec.step else 1,
            help=spec.description,
            key=key
        )
    elif spec.param_type == float:
        return st.number_input(
            label=spec.display_name,
            min_value=float(spec.min_value) if spec.min_value is not None else None,
            max_value=float(spec.max_value) if spec.max_value is not None else None,
            value=float(spec.default),
            step=float(spec.step) if spec.step else 0.01,
            format="%.3f",
            help=spec.description,
            key=key
        )
    elif spec.param_type == bool:
        return st.checkbox(
            label=spec.display_name,
            value=bool(spec.default),
            help=spec.description,
            key=key
        )
    elif spec.param_type == str and spec.options:
        return st.selectbox(
            label=spec.display_name,
            options=spec.options,
            index=spec.options.index(spec.default) if spec.default in spec.options else 0,
            help=spec.description,
            key=key
        )
    else:
        return st.text_input(
            label=spec.display_name,
            value=str(spec.default),
            help=spec.description,
            key=key
        )