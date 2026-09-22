"""
SOI (Set of Interest) UI Components.

This module provides UI components for displaying and managing saved Sets
of Interest (SOIs) in the Streamlit interface.
"""

import streamlit as st
import pandas as pd


def render_saved_sois_view() -> None:
    """
    Render the view of saved SOIs in the Working Pareto panel.

    This function displays all stored SOIs (from the analysis session or
    from session state) in expandable cards, showing their metadata
    (name, size, lens, method, group, creation time). Each SOI can be
    loaded as the active sub‑dataset or deleted.
    """
    # Retrieve SOI registry from the analysis session if available
    analysis_session = st.session_state.get("analysis_session")
    if analysis_session and hasattr(analysis_session, "soi_registry"):
        soi_registry = analysis_session.soi_registry
        saved_sois = soi_registry.list_all()
    else:
        # Fallback: use st.session_state["saved_sois"]
        saved_sois = st.session_state.get("saved_sois", [])

    if not saved_sois:
        st.caption(
            "No Sets of Interest (SOIs) saved yet. "
            "Save a set using the active Lens controls."
        )
        return

    # Display each saved SOI in an expander
    for soi in saved_sois:
        with st.expander(
            f"**{soi.name}** [{soi.size} solutions] · {soi.lens_name}",
            expanded=False,
        ):
            st.markdown(f"- **Lens / Method:** `{soi.lens_name}` / `{soi.method_name or 'N/A'}`")
            if soi.group:
                st.markdown(f"- **Group Filter:** `{soi.group}`")
            st.markdown(f"- **Created:** `{soi.created_at}`")

            col_load, col_del = st.columns([0.6, 0.4])
            with col_load:
                if st.button("⚡ Load as Active Sub-Dataset", key=f"btn_load_{soi.id}", use_container_width=True):
                    st.session_state["active_soi_id"] = soi.id
                    st.rerun()
            with col_del:
                if st.button("🗑️ Delete", key=f"btn_del_{soi.id}", use_container_width=True):
                    # Remove from the registry if available
                    if analysis_session and hasattr(analysis_session, "soi_registry"):
                        analysis_session.soi_registry.delete(soi.id)
                    else:
                        # Remove from session state fallback list
                        st.session_state["saved_sois"] = [
                            s for s in st.session_state["saved_sois"] if s.id != soi.id
                        ]
                    st.rerun()