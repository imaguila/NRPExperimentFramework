# Location: src/app.py
"""
Main Streamlit Application Entry Point.

This module coordinates the sidebar controls and the main workspace area.
It initialises the application state, renders the sidebar (data loading,
optimisation configuration, solver selection, analysis controls), and
displays the workspace panels (instance inspector, Pareto fronts, decision
maps, and detailed analysis).
"""

import streamlit as st
import pandas as pd

from src.domain.plugins.nrp import NRPPlugin
from src.interface.sidebar import render_sidebar
from src.interface.panels.instance_inspector import render_instance_inspector
from src.interface.panels.original_pareto import render_original_pareto
from src.interface.panels.working_pareto import render_working_pareto
from src.interface.panels.decision_maps import render_decision_maps


# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Extensible Framework for Domain-Agnostic Multi-Objective Optimization and Analysis",
    page_icon="⚡",
    layout="wide",
)

# -------------------- Active Plugin --------------------
ACTIVE_PLUGIN = NRPPlugin

# -------------------- Session State Initialisation --------------------
if "active_plugin" not in st.session_state:
    st.session_state["active_plugin"] = ACTIVE_PLUGIN

if "cases" not in st.session_state:
    st.session_state["cases"] = []

if "selected_case_index" not in st.session_state:
    st.session_state["selected_case_index"] = 0

# -------------------- Main Title --------------------
st.title(f"⚡Experimentation Platform {ACTIVE_PLUGIN.get_display_name()} ")

# ============================================================
# SIDEBAR RENDERING (ONCE)
# ============================================================
model, constraints_config, objectives_config, pareto_df, execution_time, solver, problem = render_sidebar(plugin=ACTIVE_PLUGIN)

# ============================================================
# READ ANALYSIS DATAFRAME FROM SESSION STATE
# ============================================================
analysis_df = st.session_state.get("analysis_df", None)

# If no analysis_df but a pareto_df exists, initialise it
if analysis_df is None and pareto_df is not None and not pareto_df.empty:
    st.session_state["analysis_df"] = pareto_df.copy()
    analysis_df = st.session_state["analysis_df"]

# ============================================================
# WORKSPACE RENDERING (RIGHT COLUMN)
# ============================================================

# 1. Instance Inspector (only if model is available)
if model is not None:
    render_instance_inspector(model, plugin=ACTIVE_PLUGIN)

# 2. Original Pareto Front (always visible if data exists)
if pareto_df is not None and not pareto_df.empty:
    render_original_pareto(
        pareto_df=pareto_df,
        objectives_config=objectives_config,
        execution_time=st.session_state.get("last_execution_time", 0.0),
        solver=st.session_state.get("last_solver", None),
        problem=st.session_state.get("last_problem", None),
        model=model,
    )

# 3. Working Pareto (Enriched & Framed) + SOIs
if pareto_df is not None and not pareto_df.empty and analysis_df is not None and not analysis_df.empty:
    if st.session_state.get("analysis_mode_enabled", False):
        render_working_pareto(
            df=analysis_df,
            objectives_config=objectives_config,
            model=model,
        )

# 4. Decision Maps (dynamic maps)
if pareto_df is not None and not pareto_df.empty and analysis_df is not None and not analysis_df.empty:
    if st.session_state.get("analysis_mode_enabled", False):
        render_decision_maps(df=analysis_df, show_ids=False)

# 5. CSS Detailed Analysis (if checkbox is enabled)
if pareto_df is not None and not pareto_df.empty and analysis_df is not None and not analysis_df.empty:
    if st.session_state.get("detailed_analysis_enabled", False):
        from src.interface.panels.css_analysis import render_css_analysis
        render_css_analysis(df=analysis_df, model=st.session_state.get("selected_model"))

# -------------------- Initial Message --------------------
if model is None and pareto_df is None:
    st.info(f"👈 Start by loading or generating a {ACTIVE_PLUGIN.get_item_label()} case in the sidebar.")