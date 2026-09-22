# Location: src/interface/panels/decision_maps.py
"""
Decision Maps Panel with multiple dynamic maps and additional views.

This module provides an interactive panel for visualising solution spaces
using scatter plots (2D, 3D, bubble) and additional maps such as violin/box
plots and parallel coordinates. Users can create, reset, and remove dynamic
map panels, each with custom axis and colour selections.
"""

import streamlit as st
import pandas as pd

# Import centralised visualisation functions
from src.interface.components.visualization import (
    render_scatter,
    render_scatter_plot,
    render_coordinated_maps,
    render_distribution,
    render_parallel_coordinates,
    infer_lens_color_column,
)


def render_decision_maps(df: pd.DataFrame, show_ids: bool = False) -> None:
    """
    Render the Decision Maps panel with dynamic and additional maps.

    The panel includes:
        - Dynamic scatter and bubble maps (users can add/remove panels).
        - Coordinated maps (two side‑by‑side 2D scatter plots sharing the X axis).
        - Additional maps: violin/box plots and parallel coordinates.

    Args:
        df: The current DataFrame containing the solutions.
        show_ids: Whether to show solution IDs on hover/labels (global default).
    """
    # Get numeric dimensions
    excluded = {"id", "selected_ids", "selected_ids_str", "is_feasible", "scope", "Selected Items"}
    num_cols = [
        c for c in df.columns
        if str(c).lower() not in excluded
        and not str(c).startswith(("req_", "x_", "var_", "item_"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not num_cols:
        st.info("No numeric dimensions available for visualisation.")
        return

    # =========================================================
    # MAIN BLOCK: DECISION MAPS (Dynamic + Additional)
    # =========================================================
    with st.expander("🗺️ Decision Maps", expanded=True):
        # --- 1. Dynamic Maps (Scatter & Bubble) ---
        if "maps" not in st.session_state or st.session_state.maps is None:
            st.session_state.maps = []

        if not st.session_state.maps and len(num_cols) >= 2:
            st.session_state.maps = [{
                "id": 1,
                "x": num_cols[0],
                "y": num_cols[1],
                "z": "None",
                "color": "None"
            }]

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Maps", use_container_width=True, key="btn_reset_maps"):
                st.session_state.maps = [{
                    "id": 1,
                    "x": num_cols[0],
                    "y": num_cols[1] if len(num_cols) > 1 else num_cols[0],
                    "z": "None",
                    "color": "None"
                }]
                st.rerun()
        with col2:
            if st.button("➕ New Map", use_container_width=True, key="btn_new_map"):
                new_id = max([m.get("id", 0) for m in st.session_state.maps]) + 1
                st.session_state.maps.append({
                    "id": new_id,
                    "x": num_cols[0],
                    "y": num_cols[1] if len(num_cols) > 1 else num_cols[0],
                    "z": "None",
                    "color": "None"
                })
                st.rerun()

        st.caption(f"Active maps: **{len(st.session_state.maps)}**")

        # Render each dynamic map
        for idx, map_item in enumerate(list(st.session_state.maps)):
            map_id = map_item.get("id", idx + 1)
            default_x = map_item.get("x", num_cols[0])
            default_y = map_item.get("y", num_cols[1] if len(num_cols) > 1 else num_cols[0])
            default_z = map_item.get("z", "None")
            default_color = map_item.get("color", "None")

            with st.expander(f"🗺️ Map Panel #{idx + 1}", expanded=True):
                col_tabs, col_del = st.columns([0.88, 0.12])
                with col_del:
                    if len(st.session_state.maps) > 1:
                        if st.button("🗑️ Remove", key=f"remove_map_{map_id}", use_container_width=True):
                            st.session_state.maps = [m for m in st.session_state.maps if m.get("id") != map_id]
                            st.rerun()

                with col_tabs:
                    chart_type = st.radio(
                        "Chart Type",
                        ["Scatter Map", "Bubble Chart"],
                        horizontal=True,
                        key=f"map_type_radio_{map_id}",
                    )

                st.markdown("---")

                # Automatically infer a colour column if available
                auto_color = infer_lens_color_column(df)
                if auto_color is None:
                    auto_color = "None"

                # --- SCATTER MAP ---
                if chart_type == "Scatter Map":
                    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 0.8, 0.8])
                    x_dim = c1.selectbox("X Axis", num_cols, index=num_cols.index(default_x) if default_x in num_cols else 0, key=f"scatter_x_{map_id}")
                    y_dim = c2.selectbox("Y Axis", num_cols, index=num_cols.index(default_y) if default_y in num_cols else 0, key=f"scatter_y_{map_id}")

                    z_options = ["None"] + [d for d in num_cols if d not in [x_dim, y_dim]]
                    z_idx = z_options.index(default_z) if default_z in z_options else 0
                    z_dim = c3.selectbox("Z Axis", z_options, index=z_idx, key=f"scatter_z_{map_id}")

                    color_options = ["None"] + [c for c in num_cols if c not in [x_dim, y_dim, z_dim]]
                    # Add special lens columns
                    for special in ["group_label", "cluster_str", "domain_match_count", "preference_score", "efficiency_score"]:
                        if special in df.columns and special not in color_options:
                            color_options.append(special)

                    if default_color not in color_options and auto_color in color_options:
                        default_color = auto_color

                    color_idx = color_options.index(default_color) if default_color in color_options else 0
                    color_dim = c4.selectbox("Color Axis", color_options, index=color_idx, key=f"scatter_color_{map_id}")
                    active_color = None if color_dim == "None" else color_dim

                    use_3d = c5.checkbox("Enable 3D", value=False, key=f"scatter_3d_{map_id}", disabled=(z_dim == "None"))
                    show_ids_local = c6.checkbox("Show IDs", value=show_ids, key=f"scatter_show_ids_{map_id}")

                    # Use centralised functions
                    if use_3d and z_dim != "None":
                        render_scatter_plot(
                            df,
                            x=x_dim,
                            y=y_dim,
                            z=z_dim,
                            color=active_color,
                            show_ids=show_ids_local,
                            use_3d=True,
                            key=f"scatter_3d_{map_id}"
                        )
                    elif z_dim != "None":
                        render_coordinated_maps(
                            df,
                            x=x_dim,
                            y=y_dim,
                            z=z_dim,
                            key_prefix=f"coord_{map_id}",
                            show_ids=show_ids_local,
                            color=active_color
                        )
                    else:
                        render_scatter(
                            df,
                            x=x_dim,
                            y=y_dim,
                            color=active_color,
                            show_ids=show_ids_local,
                            key=f"scatter_{map_id}"
                        )

                # --- BUBBLE CHART ---
                elif chart_type == "Bubble Chart":
                    c1, c2, c3, c4 = st.columns(4)
                    x_dim = c1.selectbox("X Axis", num_cols, index=num_cols.index(default_x) if default_x in num_cols else 0, key=f"bubble_x_{map_id}")
                    y_dim = c2.selectbox("Y Axis", num_cols, index=num_cols.index(default_y) if default_y in num_cols else 0, key=f"bubble_y_{map_id}")
                    size_dim = c3.selectbox("Bubble Size", num_cols, key=f"bubble_size_{map_id}")

                    color_options = ["None"] + [c for c in num_cols if c not in [x_dim, y_dim, size_dim]]
                    for special in ["group_label", "cluster_str", "domain_match_count", "preference_score", "efficiency_score"]:
                        if special in df.columns and special not in color_options:
                            color_options.append(special)

                    if default_color not in color_options and auto_color in color_options:
                        default_color = auto_color

                    color_idx = color_options.index(default_color) if default_color in color_options else 0
                    color_dim = c4.selectbox("Color Axis", color_options, index=color_idx, key=f"bubble_color_{map_id}")
                    active_color = None if color_dim == "None" else color_dim

                    show_ids_local = st.checkbox("Show IDs", value=show_ids, key=f"bubble_show_ids_{map_id}")

                    # Use render_scatter with the size parameter
                    render_scatter(
                        df,
                        x=x_dim,
                        y=y_dim,
                        size=size_dim,
                        color=active_color,
                        show_ids=show_ids_local,
                        key=f"bubble_{map_id}"
                    )

        # --- 2. Additional Maps (Violin & Parallel) ---
        st.divider()
        with st.expander("📈 Additional Maps (Violin & Parallel)", expanded=False):
            additional_view = st.radio(
                "Select View",
                ["Violin / Distribution", "Parallel Coordinates"],
                horizontal=True,
                key="additional_maps_radio"
            )

            if additional_view == "Violin / Distribution":
                c1, c2 = st.columns(2)
                target_dim = c1.selectbox("Dimension", num_cols, key="violin_dim")
                plot_style = c2.radio("Style", ["Violin", "Box"], horizontal=True, key="violin_style")

                # Use centralised render_distribution
                render_distribution(
                    df,
                    metric=target_dim,
                    mode=plot_style,
                    color=None,  # No group colouring in this panel
                    key=f"violin_box_{target_dim}"
                )

            elif additional_view == "Parallel Coordinates":
                selected_dims = st.multiselect(
                    "Dimensions to include",
                    num_cols,
                    default=num_cols[:min(5, len(num_cols))],
                    key="parallel_dims"
                )
                if len(selected_dims) >= 2:
                    # Use centralised render_parallel_coordinates
                    render_parallel_coordinates(
                        df,
                        dimensions=selected_dims,
                        key="parallel_coords"
                    )