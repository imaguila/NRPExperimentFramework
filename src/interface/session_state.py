"""
Centralised Streamlit session-state reset utilities.

This module provides a single source of truth for clearing the
different groups of values stored in st.session_state.
"""

from typing import Iterable

import streamlit as st


# ------------------------------------------------------------------
# Current optimisation and analysis result
# ------------------------------------------------------------------

ANALYSIS_STATE_KEYS = (
    "last_pareto_df",
    "last_pareto_front",
    "pareto_df",
    "pareto_front",
    "last_solver",
    "last_problem",
    "last_execution_time",
    "analysis_df",
    "analysis_df_after_lens",
    "analysis_session",
    "analysis_mode_enabled",
    "detailed_analysis_enabled",
    "enriched_df",
    "calculated_indicators",
    "selected_indicators_multiselect",
    "enrichment_just_calculated",
    "active_soi_id",
    "last_solver_name",
    "last_solver_params",
    "maps",
    "external_objectives_config",
)


# ------------------------------------------------------------------
# User-saved workspace collections
# ------------------------------------------------------------------

SAVED_COLLECTION_STATE_KEYS = (
    "saved_pareto_fronts",
    "saved_sois",
    "pareto_run_counter",
)


# ------------------------------------------------------------------
# Loaded or generated optimisation case
# ------------------------------------------------------------------

LOADED_CASE_STATE_KEYS = (
    "last_model",
    "base_model",
    "selected_model",
    "processed_models",
    "generated_model",
    "raw_json_data",
    "is_synthetic",
    "last_uploaded_file_hash",
    "last_selected_variant_idx",
    "subproblem_variant_select",
)


# ------------------------------------------------------------------
# Data-source and synthetic-generator configuration
# ------------------------------------------------------------------

DATA_SOURCE_STATE_KEYS = (
    "source_mode",
)

SYNTHETIC_GENERATOR_STATE_KEYS = (
    "synthetic_val_deps",
    "enable_precedences",
    "prec_density",
)


def _clear_keys(keys: Iterable[str]) -> None:
    """Remove the specified keys from Streamlit session state."""
    for key in keys:
        st.session_state.pop(key, None)


def reset_analysis_state() -> None:
    """
    Clear the current optimisation result and derived analysis state.

    User-saved Pareto fronts and SOIs are preserved.
    The currently loaded or generated model is also preserved.
    """
    _clear_keys(ANALYSIS_STATE_KEYS)


def reset_loaded_case_state() -> None:
    """
    Clear the loaded case and the current results derived from it.

    User-saved Pareto fronts and SOIs are preserved so that results from
    different cases or configurations can be compared within the same
    Streamlit session.
    """
    reset_analysis_state()
    _clear_keys(LOADED_CASE_STATE_KEYS)


def reset_saved_collections() -> None:
    """
    Explicitly remove Pareto fronts and SOIs saved by the user.

    This operation should only be called by a dedicated UI action such as
    'Clear Saved Results' or by a complete workspace reset.
    """
    _clear_keys(SAVED_COLLECTION_STATE_KEYS)


def reset_workspace_state() -> None:
    """
    Clear the complete application workspace.

    This removes current results, loaded models, generator configuration,
    and user-saved collections.
    """
    reset_loaded_case_state()
    reset_saved_collections()
    _clear_keys(SYNTHETIC_GENERATOR_STATE_KEYS)
    _clear_keys(DATA_SOURCE_STATE_KEYS)