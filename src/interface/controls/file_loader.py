# Location: src/interface/controls/file_loader.py

"""
File Loader Subpanel for JSON cases.

This Streamlit component collects JSON loading and preprocessing options,
delegates the computational operation to the application layer, and
allows the user to select an exclusion-generated model variant.

The component does not implement JSON extraction or preprocessing logic.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

import streamlit as st

from src.application.use_cases.load_json_case import (
    inspect_json_case,
    load_json_case,
)
from src.domain.core.model import OptimizationModel
from src.domain.plugins.base_domain import DomainPlugin
from src.interface.session_state import reset_analysis_state
from src.application.optimization_limits import (
    ALL_SUBPROBLEMS_OPTION,
    MAX_EXCLUSIONS_FOR_ALL_SUBPROBLEMS,
)


def _clear_loaded_json_state() -> None:
    """
    Clear state associated with the currently loaded JSON instance.

    Optimisation and analysis result cleanup is performed separately by
    reset_analysis_state().
    """
    keys_to_clear = (
        "base_model",
        "selected_model",
        "processed_models",
        "optimization_models",
        "optimization_scope",
        "raw_json_data",
        "is_synthetic",
        "last_selected_variant_idx",
        "last_selected_variant",
    )

    for key in keys_to_clear:
        st.session_state.pop(key, None)


def _compute_file_hash(
    file_content: bytes,
) -> str:
    """
    Compute a stable SHA-256 hash for uploaded file contents.
    """
    return hashlib.sha256(file_content).hexdigest()


def _render_aggregation_controls(
    model: OptimizationModel,
    plugin: DomainPlugin,
) -> Dict[str, str]:
    """
    Render controls for overriding multivalued aggregation rules.

    Args:
        model:
            Inspected model loaded with its default aggregation rules.

        plugin:
            Active domain plugin.

    Returns:
        Aggregation-rule overrides indexed by attribute name.
    """
    multivalued_attributes = [
        attribute_name
        for attribute_name, attribute_definition
        in model.attribute_definitions.items()
        if getattr(
            attribute_definition,
            "type",
            "scalar",
        ) == "multivalued"
    ]

    if not multivalued_attributes:
        return {}

    aggregation_options = list(
        plugin.get_supported_aggregations()
    )

    if not aggregation_options:
        return {}

    st.caption(
        "⚙️ **Evaluator Aggregation Operators**"
    )

    use_default_aggregation = st.checkbox(
        "Use default values for Aggregation",
        value=True,
        key="use_default_agg_cb",
    )

    if use_default_aggregation:
        return {}

    custom_aggregations: Dict[str, str] = {}

    for attribute_name in multivalued_attributes:
        attribute_definition = (
            model.attribute_definitions[
                attribute_name
            ]
        )

        default_aggregation = getattr(
            attribute_definition,
            "aggregation_rule",
            "weighted_mean",
        )

        default_index = (
            aggregation_options.index(
                default_aggregation
            )
            if default_aggregation
            in aggregation_options
            else 0
        )

        custom_aggregations[
            attribute_name
        ] = st.selectbox(
            f"Operator for '{attribute_name}':",
            options=aggregation_options,
            index=default_index,
            key=f"agg_{attribute_name}",
        )

    return custom_aggregations


def _render_coupling_controls(
    model: OptimizationModel,
    plugin: DomainPlugin,
) -> Dict[str, str]:
    """
    Render controls for overriding attribute coupling strategies.

    Args:
        model:
            Inspected model whose coupling relationships are examined.

        plugin:
            Active domain plugin.

    Returns:
        Coupling-rule overrides indexed by attribute name.
    """
    has_couplings = bool(
        model.relationships.get(
            "couplings",
            [],
        )
    )

    if not has_couplings:
        return {}

    coupling_options = list(
        plugin.get_supported_couplings()
    )

    if not coupling_options:
        return {}

    st.caption(
        "🔗 **Coupling Strategy per Attribute**"
    )

    use_default_coupling = st.checkbox(
        "Use default values for Coupling",
        value=True,
        key="use_default_coupling_cb",
    )

    if use_default_coupling:
        return {}

    read_attributes = (
        plugin.get_read_attributes()
        if hasattr(
            plugin,
            "get_read_attributes",
        )
        else {}
    )

    custom_couplings: Dict[str, str] = {}

    for attribute_name in (
        model.get_available_attributes()
    ):
        attribute_definition = read_attributes.get(
            attribute_name
        )

        if attribute_definition is None:
            attribute_definition = (
                model.attribute_definitions.get(
                    attribute_name
                )
            )

        default_coupling = getattr(
            attribute_definition,
            "coupling_rule",
            "sum",
        )

        default_index = (
            coupling_options.index(
                default_coupling
            )
            if default_coupling
            in coupling_options
            else 0
        )

        custom_couplings[
            attribute_name
        ] = st.selectbox(
            (
                "Coupling strategy for "
                f"'{attribute_name}':"
            ),
            options=coupling_options,
            index=default_index,
            key=f"coupling_{attribute_name}",
        )

    return custom_couplings


def _select_model_variant(
    processed_models: list[OptimizationModel],
    base_model: OptimizationModel,
    plugin: DomainPlugin,
) -> OptimizationModel:
    """
    Select one exclusion-generated model variant or configure the execution
    of all variants.

    The returned model is used as the reference model for rendering the
    formulation controls. The actual models to optimise are stored in
    session_state["optimization_models"].
    """
    if not processed_models:
        raise ValueError(
            "At least one processed model is required."
        )

    if len(processed_models) == 1:
        selected_model = processed_models[0]

        if not getattr(selected_model, "branch_name", None):
            selected_model.branch_name = "Base"

        st.session_state["last_selected_variant_idx"] = 0
        st.session_state["optimization_scope"] = "single"
        st.session_state["optimization_models"] = [selected_model]

        return selected_model

    source_exclusions = (
        getattr(base_model, "relationships", {}) or {}
    ).get("exclusions", []) or []

    exclusion_count = len(source_exclusions)

    st.warning(
        "⚠️ Exclusions detected: "
        f"{exclusion_count} exclusion(s) generated "
        f"{len(processed_models)} optimisation subproblems."
    )

    branch_names = [
        getattr(
            processed_model,
            "branch_name",
            f"Variant {index + 1}",
        )
        for index, processed_model in enumerate(processed_models)
    ]

    allow_all_subproblems = (
        exclusion_count
        <= MAX_EXCLUSIONS_FOR_ALL_SUBPROBLEMS
    )

    options = list(range(len(processed_models)))

    if allow_all_subproblems:
        options.append(ALL_SUBPROBLEMS_OPTION)
    else:
        st.info(
            "The complete exclusion-decomposed problem cannot be executed "
            f"in a single run because it contains {exclusion_count} "
            "exclusions. The current limit is "
            f"{MAX_EXCLUSIONS_FOR_ALL_SUBPROBLEMS}. "
            "Select one subproblem variant."
        )

    previous_selection = st.session_state.get(
        "last_selected_variant",
        0,
    )

    if previous_selection not in options:
        previous_selection = 0
        st.session_state["last_selected_variant"] = 0
        st.session_state.pop(
            "subproblem_variant_select",
            None,
        )

    selected_option = st.selectbox(
        "Select subproblem variant:",
        options=options,
        index=options.index(previous_selection),
        format_func=lambda option: (
            "All - Optimize all exclusion subproblems"
            if option == ALL_SUBPROBLEMS_OPTION
            else (
                f"{branch_names[option]} "
                f"({len(processed_models[option].items)} "
                f"{plugin.get_item_label_plural()})"
            )
        ),
        key="subproblem_variant_select",
    )

    if previous_selection != selected_option:
        reset_analysis_state()

        st.session_state["last_selected_variant"] = (
            selected_option
        )
        st.rerun()

    if selected_option == ALL_SUBPROBLEMS_OPTION:
        st.session_state["optimization_scope"] = "all"
        st.session_state["optimization_models"] = list(
            processed_models
        )
        st.session_state["last_selected_variant_idx"] = None

        st.info(
            "ℹ️ All exclusion subproblems will be optimised. "
            "Their partial Pareto fronts will be combined and "
            "filtered by dominance."
        )

        # Reference model used only to render common formulation controls.
        return processed_models[0]

    selected_index = int(selected_option)
    selected_model = processed_models[selected_index]

    st.session_state["optimization_scope"] = "single"
    st.session_state["optimization_models"] = [
        selected_model
    ]
    st.session_state["last_selected_variant_idx"] = (
        selected_index
    )

    return selected_model


def render_file_loader_subpanel(
    plugin: DomainPlugin,
) -> Tuple[
    Optional[OptimizationModel],
    Dict[str, Any],
]:
    """
    Render the JSON file loader and preprocessing controls.

    The interface performs three presentation responsibilities:

    1. Collect the uploaded file.
    2. Collect aggregation and coupling options.
    3. Select and display a preprocessed model variant.

    JSON parsing, model loading, and preprocessing are delegated to the
    application layer.

    Args:
        plugin:
            Active domain plugin.

    Returns:
        A tuple containing:

        - Selected preprocessed model, or None.
        - Original JSON dictionary, or an empty dictionary.
    """
    uploaded_file = st.file_uploader(
        f"Upload {plugin.get_item_label()} Case JSON instance",
        type=["json"],
        key="json_file_uploader",
    )

    if uploaded_file is None:
        _clear_loaded_json_state()
        return None, {}

    file_content = uploaded_file.getvalue()

    if not file_content:
        _clear_loaded_json_state()

        st.error(
            "The uploaded JSON file is empty."
        )

        return None, {}

    current_file_hash = _compute_file_hash(
        file_content
    )

    previous_file_hash = st.session_state.get(
        "last_uploaded_file_hash"
    )

    if previous_file_hash != current_file_hash:
        reset_analysis_state()
        _clear_loaded_json_state()

        st.session_state[
            "last_uploaded_file_hash"
        ] = current_file_hash

        st.session_state.pop(
            "subproblem_variant_select",
            None,
        )

    try:
        raw_json_text = file_content.decode(
            "utf-8"
        )

    except UnicodeDecodeError:
        _clear_loaded_json_state()

        st.error(
            "The uploaded file is not a valid UTF-8 JSON file."
        )

        return None, {}

    # --------------------------------------------------
    # Inspect the instance for rendering configuration
    # controls.
    # --------------------------------------------------

    try:
        inspection = inspect_json_case(
            json_input=raw_json_text,
            plugin=plugin,
        )

    except json.JSONDecodeError as exc:
        _clear_loaded_json_state()

        st.error(
            "The uploaded file does not contain valid JSON. "
            f"Line {exc.lineno}, column {exc.colno}: {exc.msg}"
        )

        return None, {}

    except (TypeError, ValueError, KeyError) as exc:
        _clear_loaded_json_state()

        st.error(
            "The JSON instance could not be inspected: "
            f"{type(exc).__name__}: {exc}"
        )

        return None, {}

    custom_aggregations = (
        _render_aggregation_controls(
            model=inspection.inspected_model,
            plugin=plugin,
        )
    )

    custom_couplings = (
        _render_coupling_controls(
            model=inspection.inspected_model,
            plugin=plugin,
        )
    )

    # --------------------------------------------------
    # Execute headless loading and preprocessing use case.
    # --------------------------------------------------

    try:
        result = load_json_case(
            json_input=raw_json_text,
            plugin=plugin,
            aggregation_overrides=custom_aggregations,
            coupling_overrides=custom_couplings,
        )

    except json.JSONDecodeError as exc:
        _clear_loaded_json_state()

        st.error(
            "The uploaded file does not contain valid JSON. "
            f"Line {exc.lineno}, column {exc.colno}: {exc.msg}"
        )

        return None, {}

    except (
        TypeError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as exc:
        _clear_loaded_json_state()

        st.error(
            "The JSON instance could not be loaded or "
            "preprocessed. "
            f"{type(exc).__name__}: {exc}"
        )

        return None, {}

    selected_model = _select_model_variant(
        processed_models=result.processed_models,
        base_model=result.base_model,
        plugin=plugin,
    )

    # --------------------------------------------------
    # Store current interface state.
    # --------------------------------------------------

    st.session_state[
        "base_model"
    ] = result.base_model

    st.session_state[
        "processed_models"
    ] = result.processed_models

    st.session_state[
        "selected_model"
    ] = selected_model

    st.session_state[
        "raw_json_data"
    ] = result.raw_data

    st.session_state[
        "is_synthetic"
    ] = False

    optimization_scope = st.session_state.get(
        "optimization_scope",
        "single",
    )

    if optimization_scope == "all":
        st.success(
            f"Loaded: **{selected_model.name}** | "
            f"Scope: **all {len(result.processed_models)} subproblems**"
        )
    else:
        branch_name = getattr(
            selected_model,
            "branch_name",
            "Base",
        )

        st.success(
            f"Loaded: **{selected_model.name}** | "
            f"Branch: **{branch_name}** | "
            f"Active {plugin.get_item_label_plural()}: "
            f"**{len(selected_model.items)}**"
        )

    return selected_model, result.raw_data