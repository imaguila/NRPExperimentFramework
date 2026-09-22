# Location: src/interface/panels/original_pareto.py
"""
Original Pareto Front Panel.

This module provides the UI for displaying the unmodified Pareto front
with multiple views: overview, table, maps, saved fronts, and comparison.
"""

import streamlit as st
import pandas as pd

from src.interface.components.visualization import (
    render_summary_metrics,
    render_dataset_table,
    render_scatter_plot,
    render_coordinated_maps,
)
from src.interface.components.soi_panel import render_saved_sois_view
from src.solving.core.problem import OptimizationProblem
from src.domain.core.paretofront import ParetoFront


def render_export_section(df: pd.DataFrame, key_prefix: str = "original") -> None:
    """
    Render a single CSV export control with a unique key.

    Args:
        df: The DataFrame to export.
        key_prefix: Prefix for the download button key.
    """
    st.download_button(
        label="📊 Export Current Set (.csv)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="current_set.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{key_prefix}_btn_export_summary_csv",
    )


def render_original_pareto(
    pareto_df: pd.DataFrame,
    objectives_config,
    execution_time,
    solver,
    problem,
    model=None
) -> None:
    """
    Render the Original Pareto Front panel with multiple views.

    Args:
        pareto_df: The Pareto front DataFrame.
        objectives_config: Mapping of objective names to directions.
        execution_time: Optimisation execution time (seconds).
        solver: The solver instance used.
        problem: The OptimizationProblem instance.
        model: Optional model for decision variable count.
    """
    with st.expander("📊 Original Pareto Front", expanded=True):
        # Radio button for view selection
        view_mode = st.radio(
            "Select View:",
            options=["📊 Overview", "📋 Pareto Front Table", "🗺️ Maps", "📦 Saved Pareto Fronts", "📊 Comparison"],
            horizontal=True,
            key="original_pareto_view"
        )

        # 1. Overview View
        if view_mode == "📊 Overview":
            if pareto_df is not None and not pareto_df.empty:
                obj_cols = set(objectives_config.keys())

                # Get decision variable count from model or infer from columns
                if model is not None and hasattr(model, "items"):
                    num_decision_vars = len(model.items)
                else:
                    decision_cols = [col for col in pareto_df.columns if col.startswith("req_")]
                    num_decision_vars = len(decision_cols)

                solver_name = solver.get_name() if (solver and hasattr(solver, "get_name")) else "N/A"
                exec_time_str = f"{execution_time:.2f}s" if execution_time > 0 else "N/A"

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Solutions", len(pareto_df))
                c2.metric("Decision Vars (Reqs)", num_decision_vars)

                obj_names = sorted(obj_cols)
                obj_label = f"Objectives ({', '.join(obj_names)})" if obj_names else "Objectives"
                c3.metric(obj_label, len(obj_cols))

                c4.metric("Execution Time", exec_time_str)
                c5.metric("Solver", solver_name)

            st.divider()
            render_export_section(pareto_df, key_prefix="original")

        # 2. Pareto Front Table
        elif view_mode == "📋 Pareto Front Table":
            render_dataset_table(pareto_df, objectives_config)

        # 3. Maps
        elif view_mode == "🗺️ Maps":
            render_pareto_maps(pareto_df)

        # 4. Saved Pareto Fronts
        elif view_mode == "📦 Saved Pareto Fronts":
            render_saved_pareto_fronts_view(problem)

        # 5. Comparison
        elif view_mode == "📊 Comparison":
            from src.interface.panels.comparison_panel import render_comparison_panel

            pareto_front = st.session_state.get("last_pareto_front")
            if pareto_front is None and pareto_df is not None:
                decision_cols = [c for c in pareto_df.columns if c.startswith("req_")]
                objective_cols = [c for c in pareto_df.columns if c not in decision_cols and c not in ["id", "selected_ids", "selected_ids_str", "is_feasible"]]
                pareto_front = ParetoFront.from_dataframe(
                    df=pareto_df,
                    decision_cols=decision_cols,
                    objective_cols=objective_cols
                )

            problem_to_use = problem
            if problem_to_use is None:
                problem_to_use = st.session_state.get("last_problem")

            render_comparison_panel(pareto_front, problem_to_use)


def render_pareto_maps(df: pd.DataFrame) -> None:
    """
    Render paired charts (same X, two different Y) for the Pareto front.

    Uses centralised visualisation functions to avoid duplication.
    Allows selection of X, Y, Z (3D), and Colour axes.

    Args:
        df: The DataFrame containing the Pareto front.
    """
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

    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 0.8, 0.8])
    x_dim = c1.selectbox("X Axis", num_cols, index=0, key="pareto_map_x")
    y_dim = c2.selectbox("Y Axis", num_cols, index=1 if len(num_cols) > 1 else 0, key="pareto_map_y")
    z_options = ["None"] + [d for d in num_cols if d not in [x_dim, y_dim]]
    z_dim = c3.selectbox("Z Axis", z_options, index=0, key="pareto_map_z")
    color_options = ["None"] + num_cols
    color_dim = c4.selectbox("Color Axis", color_options, key="pareto_map_color")
    active_color = None if color_dim == "None" else color_dim
    use_3d = c5.checkbox("Enable 3D", value=False, key="pareto_map_3d", disabled=(z_dim == "None"))
    show_ids = c6.checkbox("Show IDs", value=False, key="pareto_map_show_ids")

    # 3D case
    if use_3d and z_dim != "None":
        render_scatter_plot(
            df,
            x=x_dim,
            y=y_dim,
            z=z_dim,
            color=active_color,
            show_ids=show_ids,
            use_3d=True,
            key="pareto_scatter_3d",
            height=520
        )
    elif z_dim != "None":
        # Two 2D charts: X-Y and X-Z
        render_coordinated_maps(
            df,
            x=x_dim,
            y=y_dim,
            z=z_dim,
            key_prefix="pareto_coord",
            show_ids=show_ids,
            color=active_color
        )
    else:
        # Single 2D chart, with optional alternative Y2 axis
        y2_options = [d for d in num_cols if d not in [x_dim, y_dim]]
        if y2_options:
            y2_dim = st.selectbox("Y2 Axis (Alternative)", y2_options, index=0, key="pareto_map_y2")
            col1, col2 = st.columns(2)
            with col1:
                render_scatter_plot(
                    df,
                    x=x_dim,
                    y=y_dim,
                    color=active_color,
                    show_ids=show_ids,
                    key="pareto_scatter_xy",
                    height=440
                )
            with col2:
                render_scatter_plot(
                    df,
                    x=x_dim,
                    y=y2_dim,
                    color=active_color,
                    show_ids=show_ids,
                    key="pareto_scatter_xy2",
                    height=440
                )
        else:
            render_scatter_plot(
                df,
                x=x_dim,
                y=y_dim,
                color=active_color,
                show_ids=show_ids,
                key="pareto_scatter_xy",
                height=460
            )


def render_saved_pareto_fronts_view(problem: OptimizationProblem) -> None:
    """
    Render the view of saved Pareto fronts.

    Args:
        problem: The OptimizationProblem containing the saved fronts.
    """
    saved_paretos = st.session_state.get("saved_pareto_fronts", [])

    if not saved_paretos:
        st.caption("No Pareto fronts saved yet. Use the 'Save Pareto Front' button in the sidebar after running an optimisation.")
        return

    for p in saved_paretos:
        with st.expander(f"**{p['name']}** [{len(p['df'])} solutions] · {p['solver_name']}"):
            st.markdown(f"- **Created:** `{p['created_at']}`")
            st.markdown(f"- **Execution Time:** `{p['execution_time']:.2f}s`")
            st.markdown(f"- **Solver:** `{p['solver_name']}`")

            # Display objectives and constraints if available
            if "objectives_config" in p and p["objectives_config"]:
                obj_str = ", ".join([f"{k} ({v})" for k, v in p["objectives_config"].items()])
                st.markdown(f"- **Objectives:** `{obj_str}`")

            if "constraints_config" in p and p["constraints_config"]:
                cons_str_list = []
                for attr, spec in p["constraints_config"].items():
                    if isinstance(spec, dict):
                        op = spec.get("operator", "?")
                        val = spec.get("value", "?")
                        cons_str_list.append(f"{attr} {op} {val}")
                    else:
                        cons_str_list.append(f"{attr}: {spec}")
                st.markdown(f"- **Constraints:** `{', '.join(cons_str_list)}`")

            col_load, col_del = st.columns([0.6, 0.4])

            with col_load:
                if st.button("⚡ Load as Active Front", key=f"btn_load_{p['id']}", use_container_width=True):
                    from src.domain.core.paretofront import ParetoFront

                    loaded_front = ParetoFront.from_dataframe(
                        df=p["df"],
                        decision_cols=p.get("decision_cols", []),
                        objective_cols=p.get("objective_cols", [])
                    )

                    st.session_state["last_pareto_df"] = p["df"].copy()
                    st.session_state["last_pareto_front"] = loaded_front
                    st.session_state["last_execution_time"] = p["execution_time"]
                    st.session_state["last_solver"] = None
                    st.rerun()

            with col_del:
                if st.button("🗑️ Delete", key=f"btn_del_{p['id']}", use_container_width=True):
                    st.session_state["saved_pareto_fronts"] = [
                        f for f in st.session_state["saved_pareto_fronts"] if f["id"] != p["id"]
                    ]
                    st.rerun()