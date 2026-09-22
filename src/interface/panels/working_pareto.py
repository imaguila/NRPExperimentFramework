# Location: src/interface/panels/working_pareto.py
"""
Working Pareto Panel (Enriched & Filtered).

This panel displays the enriched and framed Pareto front, offering three views:
    - Overview: summary metrics and export.
    - Current Set: detailed table of the current dataset.
    - Saved SOIs: list of stored Sets of Interest.
"""

import streamlit as st
import pandas as pd

from src.interface.components.visualization import (
    render_summary_metrics,
    render_dataset_table,
    render_export_section,
)
from src.interface.components.soi_panel import render_saved_sois_view


def render_working_pareto(df: pd.DataFrame, objectives_config, model) -> None:
    """
    Render the Working Pareto (enriched and framed) panel.

    Args:
        df: The current working DataFrame.
        objectives_config: Mapping of objective names to directions.
        model: The OptimizationModel (used for decision variable counts).
    """
    with st.expander("📊 Working Pareto (Enriched & Framed)", expanded=True):
        view_mode = st.radio(
            "Summary View",
            ["📊 Overview", "📋 Current Set", "🔖 Saved SOIs"],
            horizontal=True,
            key="working_pareto_view"
        )

        if view_mode == "📊 Overview":
            render_summary_metrics(df, objectives_config=objectives_config, selected_model=model)
            st.divider()
            render_export_section(df, key_prefix="working")

        elif view_mode == "📋 Current Set":
            render_dataset_table(df, objectives_config)

        elif view_mode == "🔖 Saved SOIs":
            render_saved_sois_view()