# Location: src/interface/controls/lenses.py
"""
Lenses Panel UI Component.

Renders control widgets for analytical lenses in a generic way.
Delegates all schema and parameter logic to the lenses themselves.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
from datetime import datetime

from src.analysis.lenses.registry import get_lens_names, get_lens


def _save_soi(
    df: pd.DataFrame,
    lens_name: str,
    params: Dict[str, Any],
    group_col: Optional[str] = None,
) -> None:
    """
    Save the current DataFrame as an SOI in the analysis session registry
    (if available) or fall back to st.session_state["saved_sois"].

    Args:
        df: The DataFrame to save.
        lens_name: Name of the lens that generated the set.
        params: Parameters used for the lens.
        group_col: Optional column name for grouping (e.g., 'group_label').
    """
    # Define the SOI class locally if not available
    try:
        from src.analysis.core.soi import SOI
    except ImportError:
        # Simple inline definition with size and created_at
        from dataclasses import dataclass, field
        from typing import List

        @dataclass
        class SOI:
            id: str
            name: str
            solution_ids: List[Any]
            lens_name: str
            method_name: Optional[str] = None
            group: Optional[str] = None
            params: Dict[str, Any] = field(default_factory=dict)
            created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            @property
            def size(self) -> int:
                return len(self.solution_ids)

    if df is None or df.empty:
        return

    st.markdown("---")
    st.markdown("##### 💾 Save Current Set as SOI")

    working_set = df.copy()
    selected_group_name = None

    if group_col and group_col in df.columns:
        groups = ["All groups"] + sorted(df[group_col].dropna().astype(str).unique().tolist())
        selected_group = st.selectbox(
            "Filter Group to Save",
            groups,
            key=f"soi_group_filter_{lens_name}",
        )
        if selected_group != "All groups":
            working_set = df[df[group_col].astype(str) == str(selected_group)].copy()
            selected_group_name = str(selected_group)

    st.caption(f"Candidate SOI size: **{len(working_set)}** solutions")

    # Get SOI registry from the analysis session if available
    analysis_session = st.session_state.get("analysis_session")
    if analysis_session and hasattr(analysis_session, "soi_registry"):
        registry = analysis_session.soi_registry
        total_sois = len(registry.list_all())
    else:
        # Fallback: use a simple list in session_state
        if "saved_sois" not in st.session_state:
            st.session_state["saved_sois"] = []
        total_sois = len(st.session_state["saved_sois"])

    group_suffix = f" ({selected_group_name})" if selected_group_name else ""
    default_name = f"{lens_name}{group_suffix} Set #{total_sois + 1}"

    soi_name = st.text_input("SOI Name", value=default_name, key=f"soi_name_input_{lens_name}")

    if st.button("💾 Save SOI", use_container_width=True, type="primary", key=f"btn_save_soi_{lens_name}"):
        solution_ids = working_set["id"].tolist() if "id" in working_set.columns else working_set.index.tolist()
        soi_id = f"soi_{total_sois + 1}_{pd.Timestamp.now().strftime('%H%M%S')}"
        method_name = params.get("method", "Exploratory")

        new_soi = SOI(
            id=soi_id,
            name=soi_name,
            solution_ids=solution_ids,
            lens_name=lens_name,
            method_name=method_name,
            group=selected_group_name,
            params=params,
        )

        # Save to the session registry if available
        if analysis_session and hasattr(analysis_session, "soi_registry"):
            registry.add(new_soi)
        else:
            # Fallback: save to simple session state list
            st.session_state["saved_sois"].append(new_soi)

        st.success(f"Saved '{soi_name}' ({len(solution_ids)} solutions)!")
        st.rerun()


def render_lenses_panel(
    df: pd.DataFrame,
    available_dimensions: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Render the analytical lenses controls in the sidebar in a generic way.

    Args:
        df: The current DataFrame.
        available_dimensions: List of available numeric dimension names.

    Returns:
        A tuple (processed_df, active_lens_name, lens_params).
    """
    if df is None or df.empty:
        st.sidebar.warning("No data available for analytical lenses.")
        return df, "None", {}

    processed_df = df.copy()
    lens_params: Dict[str, Any] = {}

    # Determine if the expander should be open (if a lens result already exists)
    is_expanded = st.session_state.get("analysis_df_after_lens") is not None

    with st.sidebar.expander("🔍 Analytical Lenses", expanded=is_expanded):
        # 1. Lens type selection
        lens_names = get_lens_names()
        previous_lens = st.session_state.get("sb_active_lens_type", "None")
        selected_lens_type = st.selectbox(
            "Select Lens Type",
            options=lens_names,
            index=0,
            key="sb_active_lens_type",
        )

        # If the lens type changes, clear previous results and saved schema
        if previous_lens != selected_lens_type:
            st.session_state["analysis_df_after_lens"] = None
            st.session_state["lens_groups"] = None
            st.session_state["analysis_df_full"] = None
            st.session_state["last_lens_schema"] = None

        # 2. If "None", do nothing
        if selected_lens_type == "None":
            st.caption("No analytical lens active.")
            if st.session_state.get("analysis_df_after_lens") is not None:
                return st.session_state["analysis_df_after_lens"], "None", {}
            return processed_df, "None", {}

        # 3. Special case: Manual Selection
        if selected_lens_type == "Manual Selection":
            if "id" not in df.columns:
                st.error("DataFrame must have an 'id' column for manual selection.")
                return df, "None", {}

            valid_ids = df["id"].dropna().astype(str).tolist()
            if not valid_ids:
                st.warning("No valid IDs found in the dataset.")
                return df, "None", {}

            selected_ids = st.multiselect(
                "Pick solutions by ID",
                options=valid_ids,
                key="manual_selection_ids",
            )

            lens_params = {"selected_ids": selected_ids}

            if st.button("Apply Manual Selection Lens", key="btn_apply_manual"):
                lens = get_lens("Manual Selection")
                if lens:
                    processed_df = lens.apply(df, lens_params)
                    st.session_state["analysis_df_after_lens"] = processed_df
                    st.session_state["analysis_df_full"] = processed_df.copy()
                    st.success(f"Applied Manual Selection lens. {len(processed_df)} solutions remaining.")
                    st.rerun()
                else:
                    st.error("Manual Selection lens not found in registry.")

            if st.session_state.get("analysis_df_after_lens") is not None:
                active_df = st.session_state["analysis_df_after_lens"]
                _save_soi(active_df, selected_lens_type, lens_params)
            else:
                st.caption("Apply the manual selection lens to enable SOI saving.")

            if st.session_state.get("analysis_df_after_lens") is not None:
                return st.session_state["analysis_df_after_lens"], "Manual Selection", lens_params
            return processed_df, "Manual Selection", lens_params

        # 4. Other lenses (generic)
        lens = get_lens(selected_lens_type)
        if lens is None:
            st.error(f"Lens '{selected_lens_type}' not found.")
            return df, "None", {}

        # --- Build dynamic context ---
        # Use the last saved schema to know which controls existed
        last_schema = st.session_state.get("last_lens_schema")
        context = {}

        if last_schema is not None:
            # Read current values from session_state for each control in the previous schema
            for item in last_schema:
                key = item.get("key")
                if key:
                    st_key = f"lens_{selected_lens_type}_{key}"
                    if st_key in st.session_state:
                        context[key] = st.session_state[st_key]

        # Add the lens type to the context (useful for Diversity)
        context["_lens_type"] = selected_lens_type

        # If Consensus, get SOIs from the session registry
        if selected_lens_type == "Consensus":
            analysis_session = st.session_state.get("analysis_session")
            if analysis_session and hasattr(analysis_session, "soi_registry"):
                saved_sois = analysis_session.soi_registry.list_all()
            else:
                saved_sois = st.session_state.get("saved_sois", [])
            context["saved_sois"] = saved_sois

        # Get the schema (now with updated context)
        schema = lens.get_schema(df, available_dimensions, context=context)

        # Store the schema for the next run
        st.session_state["last_lens_schema"] = schema

        # Render controls
        for item in schema:
            key = item.get("key")
            label = item.get("label", key)
            field_type = item.get("type")

            val_min = item.get("min", 0)
            val_max = item.get("max", 100)
            val_default = item.get("default", val_min)
            options = item.get("options", [])
            st_key = f"lens_{selected_lens_type}_{key}"

            # If the value already exists in session_state, use it as default
            if st_key in st.session_state:
                val_default = st.session_state[st_key]

            if field_type == "select":
                default_idx = options.index(val_default) if val_default in options else 0
                lens_params[key] = st.selectbox(
                    label,
                    options=options,
                    index=default_idx,
                    key=st_key,
                )
            elif field_type == "multiselect":
                if not options:
                    st.warning(f"No options available for {label}")
                    lens_params[key] = []
                else:
                    if not isinstance(val_default, list):
                        val_default = [val_default] if val_default else []
                    lens_params[key] = st.multiselect(
                        label,
                        options=options,
                        default=val_default,
                        key=st_key,
                    )
            elif field_type == "checkbox":
                lens_params[key] = st.checkbox(
                    label,
                    value=bool(val_default),
                    key=st_key,
                )
            elif field_type == "slider":
                lens_params[key] = st.slider(
                    label,
                    min_value=int(val_min),
                    max_value=int(val_max),
                    value=int(val_default),
                    key=st_key,
                )
            elif field_type == "slider_float":
                lens_params[key] = st.slider(
                    label,
                    min_value=float(val_min),
                    max_value=float(val_max),
                    value=float(val_default),
                    step=float(item.get("step", 0.05)),
                    key=st_key,
                )
            # If there are info-only fields, display them as text
            elif field_type == "info":
                st.caption(item.get("content", ""))

        # Button to apply the lens
        if st.button(f"Apply {selected_lens_type} Lens", key=f"btn_apply_{selected_lens_type}"):
            processed_df = lens.apply(df, lens_params, context=context)
            st.session_state["analysis_df_after_lens"] = processed_df
            st.session_state["analysis_df_full"] = processed_df.copy()

            # Store the list of available groups
            if "group_label" in processed_df.columns:
                groups = ["All Groups"] + sorted(processed_df["group_label"].unique().tolist())
                st.session_state["lens_groups"] = groups
            elif "cluster_str" in processed_df.columns:
                groups = ["All Clusters"] + sorted(processed_df["cluster_str"].unique().tolist())
                st.session_state["lens_groups"] = groups
            else:
                st.session_state["lens_groups"] = ["All"]

            st.success(f"Applied {selected_lens_type} lens. {len(processed_df)} solutions remaining.")
            st.rerun()

        # --- Group selector and DataFrame filtering ---
        if st.session_state.get("analysis_df_after_lens") is not None:
            current_df = st.session_state["analysis_df_after_lens"]
            groups = st.session_state.get("lens_groups", ["All"])
            selected_group = st.selectbox(
                "Select SOI Group to Display",
                options=groups,
                key="lens_group_selector"
            )
            # Apply filter
            if selected_group != "All Groups" and selected_group != "All Clusters" and selected_group != "All":
                if "group_label" in current_df.columns:
                    filtered_df = current_df[current_df["group_label"] == selected_group].copy()
                elif "cluster_str" in current_df.columns:
                    filtered_df = current_df[current_df["cluster_str"] == selected_group].copy()
                else:
                    filtered_df = current_df
                st.session_state["analysis_df_after_lens"] = filtered_df
            else:
                # Restore full DataFrame
                if "analysis_df_full" in st.session_state:
                    st.session_state["analysis_df_after_lens"] = st.session_state["analysis_df_full"].copy()

        # --- Save SOI (after applying the lens) ---
        if st.session_state.get("analysis_df_after_lens") is not None:
            active_df = st.session_state["analysis_df_after_lens"]
            # Detect group column for saving
            group_col = None
            if "group_label" in active_df.columns:
                group_col = "group_label"
            elif "cluster_str" in active_df.columns:
                group_col = "cluster_str"
            elif "group_base" in active_df.columns:
                group_col = "group_base"
            _save_soi(active_df, selected_lens_type, lens_params, group_col)
        else:
            st.caption("Apply a lens to enable SOI saving.")

    # Return the final DataFrame (the one in session_state)
    if st.session_state.get("analysis_df_after_lens") is not None:
        return st.session_state["analysis_df_after_lens"], selected_lens_type, lens_params
    return processed_df, selected_lens_type, lens_params