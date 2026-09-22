"""
Data Loader Panel for Streamlit Control Sidebar.

Handles file uploading for JSON and CSV files, as well as synthetic case
generation. The user can select between:

- Synthetic Case Generator
- Upload Case File (JSON)
- Upload Pareto Front (CSV)
"""

from typing import Any, Dict, Optional, Tuple

import streamlit as st

from src.domain.core.model import OptimizationModel
from src.domain.core.paretofront import load_pareto_csv
from src.domain.plugins.base_domain import DomainPlugin
from src.interface.controls.file_loader import render_file_loader_subpanel
from src.interface.controls.synthetic_generator import (
    render_synthetic_generator_panel,
)
from src.interface.session_state import reset_analysis_state


def render_data_loader_panel(
    plugin: DomainPlugin,
) -> Tuple[
    Optional[OptimizationModel],
    Optional[Dict[str, Any]],
]:
    """
    Render the data-source selection and loading panel.

    Args:
        plugin:
            Active domain plugin.

    Returns:
        A tuple containing:

        - The loaded or generated OptimizationModel, when available.
        - Additional source metadata.

        For an imported Pareto-front CSV, no OptimizationModel is available,
        so the first tuple value is None and the metadata dictionary contains
        the ParetoFront and its DataFrame.
    """
    with st.expander("📍 Data Source", expanded=True):
        source_option = st.radio(
            "Choose Data Source:",
            options=[
                "Synthetic Case Generator",
                "Upload Case File (JSON)",
                "Upload Pareto Front (CSV)",
            ],
            key="data_source_radio",
            horizontal=True,
        )

        st.markdown("---")

        # --------------------------------------------------
        # Detect a change of data source
        # --------------------------------------------------

        if "last_source_option" not in st.session_state:
            st.session_state["last_source_option"] = source_option

        if st.session_state["last_source_option"] != source_option:
            st.session_state["last_source_option"] = source_option

            # Previous optimization and analysis results are no longer valid.
            reset_analysis_state()

            st.rerun()

        # --------------------------------------------------
        # Option 1: Synthetic Case Generator
        # --------------------------------------------------

        if source_option == "Synthetic Case Generator":
            return render_synthetic_generator_panel(
                plugin=plugin,
            )

        # --------------------------------------------------
        # Option 2: Upload JSON Case File
        # --------------------------------------------------

        if source_option == "Upload Case File (JSON)":
            return render_file_loader_subpanel(
                plugin=plugin,
            )

        # --------------------------------------------------
        # Option 3: Upload Pareto Front CSV
        # --------------------------------------------------

        if source_option == "Upload Pareto Front (CSV)":
            uploaded_file = st.file_uploader(
                "Upload Pareto Front CSV",
                type=["csv"],
                key="pareto_csv_uploader",
            )

            if uploaded_file is None:
                return None, None

            pareto_front, pareto_df = load_pareto_csv(
                filepath_or_buffer=uploaded_file,
                objective_directions=st.session_state.get(
                    "external_objectives_config"
                ),
            )

            # Current imported Pareto-front state.
            st.session_state["pareto_front"] = pareto_front
            st.session_state["pareto_df"] = pareto_df

            # Keep the same result available through the common last-result
            # keys used by the rest of the application.
            st.session_state["last_pareto_front"] = pareto_front
            st.session_state["last_pareto_df"] = pareto_df

            st.session_state["analysis_df"] = pareto_df.copy()
            st.session_state["source_mode"] = "external_pareto"

            return None, {
                "pareto_front": pareto_front,
                "df": pareto_df,
            }

    return None, None
