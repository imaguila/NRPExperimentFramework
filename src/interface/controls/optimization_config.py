# Location: src/interface/controls/optimization_config.py
"""
Optimization Problem Configuration Panel (Sidebar Control).

This module provides the UI panel for configuring optimisation constraints
and objectives on the loaded model.
"""

import streamlit as st
from typing import Dict, Tuple, List, Any

from src.domain.core.model import OptimizationModel
from src.domain.plugins.base_domain import DomainPlugin
from src.solving.core.problem import ThresholdOperator, ConstraintSpec
from src.solving.core.evaluator import SolutionEvaluator


def render_optimization_config_panel(
    selected_model: OptimizationModel,
    plugin: DomainPlugin
) -> Tuple[Dict[str, ConstraintSpec], Dict[str, str]]:
    """
    Render the optimisation configuration panel in the sidebar.

    This panel allows the user to select constraint attributes, set their
    operators and threshold values, and choose which attributes to maximise
    or minimise as objectives.

    Args:
        selected_model: The current OptimizationModel.
        plugin: The active domain plugin (for labels and supported rules).

    Returns:
        A tuple (constraints_config, objectives_config) where:
            - constraints_config: Mapping of attribute names to ConstraintSpec.
            - objectives_config: Mapping of attribute names to 'max' or 'min'.
    """
    model = getattr(selected_model, "model", selected_model)

    # Ensure attribute_definitions are objects (convert dicts if needed)
    attr_defs_clean = {}
    for k, v in model.attribute_definitions.items():
        if isinstance(v, dict):
            from src.domain.core.definitions import AttributeDefinition
            attr_defs_clean[k] = AttributeDefinition.from_dict(k, v)
        else:
            attr_defs_clean[k] = v
    model.attribute_definitions = attr_defs_clean

    if hasattr(model, "get_available_attributes"):
        available_attrs = model.get_available_attributes()
    else:
        items = getattr(model, "items", {}) or {}
        extracted = set()
        for item in items.values():
            item_attrs = getattr(item, "attributes", {})
            if isinstance(item_attrs, dict):
                extracted.update(item_attrs.keys())
        available_attrs = sorted(list(extracted))

    if not model or not available_attrs:
        return {}, {}

    constraints_config: Dict[str, ConstraintSpec] = {}
    objectives_config: Dict[str, str] = {}

    evaluator = SolutionEvaluator(model)
    items_dict = getattr(model, "items", {}) or {}
    all_item_ids = {str(k) for k in items_dict.keys()}

    # --- Constraints ---
    st.markdown("### 💰 Resource & Threshold Constraints")

    default_constraint_attrs = [available_attrs[0]] if available_attrs else []

    selected_constraint_attrs = st.multiselect(
        "Select Constraint Attributes:",
        options=available_attrs,
        default=default_constraint_attrs,
        key="config_constraint_attrs_select"
    )

    for attr in selected_constraint_attrs:
        st.markdown(f"**Attribute: `{attr}`**")

        if hasattr(model, "get_total_capacity"):
            base_capacity = model.get_total_capacity(attr)
        else:
            base_capacity = sum(
                float(getattr(item, "attributes", {}).get(attr, 0.0))
                for item in items_dict.values()
            )

        evaluated_capacity = evaluator.evaluate_attribute(all_item_ids, attr)
        total_capacity = max(evaluated_capacity, base_capacity)
        max_limit = int(total_capacity) if total_capacity > 0 else 100

        operator = st.selectbox(
            f"Operator for {attr}:",
            options=[op.value for op in ThresholdOperator],
            index=0,
            key=f"op_select_{attr}"
        )

        if operator == ThresholdOperator.BETWEEN.value:
            default_min = max(0, int(max_limit * 0.1))
            default_max = max(1, int(max_limit * 0.5))
            target_val = st.slider(
                f"Range for {attr}:",
                min_value=0,
                max_value=max_limit,
                value=(default_min, default_max),
                key=f"slider_between_{attr}"
            )
        else:
            default_val = max(1, int(max_limit * 0.4))
            target_val = st.slider(
                f"Limit for {attr}:",
                min_value=0,
                max_value=max_limit,
                value=default_val,
                key=f"slider_single_{attr}"
            )

        constraints_config[attr] = ConstraintSpec(
            operator=ThresholdOperator(operator),
            value=target_val
        )

    st.markdown("### 🎯 Objectives Configuration")

    constraint_names = set(selected_constraint_attrs)
    unconstrained_attrs = [a for a in available_attrs if a not in constraint_names]
    default_max_objs = [unconstrained_attrs[0]] if unconstrained_attrs else []

    max_selected = st.multiselect(
        "Objectives to MAXIMIZE:",
        options=available_attrs,
        default=default_max_objs,
        key="config_max_objs_select"
    )

    remaining_for_min = [a for a in available_attrs if a not in max_selected]

    min_selected = st.multiselect(
        "Objectives to MINIMIZE:",
        options=remaining_for_min,
        default=[],
        key="config_min_objs_select"
    )

    for obj in max_selected:
        objectives_config[obj] = "max"
    for obj in min_selected:
        objectives_config[obj] = "min"

    return constraints_config, objectives_config