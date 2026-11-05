# Location: src/interface/panels/css_analysis.py
"""
Candidate Solution Set (CSS) Detailed Analysis Panel.

This module provides a detailed comparison view for a selected set of solutions
(SOI). It includes:
    - A radar chart for comparative profiles (customisable metrics).
    - A heatmap showing requirement composition across solutions.
    - Stakeholder coverage and alignment analysis (radar + alignment matrix)
      for domains that support it (e.g., NRP).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Any, Optional

from src.interface.components.visualization import render_scatter  # optionally used


def render_css_analysis(df: pd.DataFrame, model: Any = None) -> None:
    """
    Render the detailed CSS analysis panel.

    The panel expects a DataFrame containing the current SOI (enriched and/or
    framed). It allows the user to select individual solutions to compare,
    and displays three tabs:
        1. Comparative Profile (radar chart of custom metrics)
        2. Requirement Composition (heatmap of selected requirements)
        3. Stakeholder Impact (coverage and alignment, if the model provides it)

    Args:
        df: The DataFrame of the current SOI.
        model: Optional OptimizationModel (used to access stakeholders and plugin).
    """
    if df is None or df.empty:
        st.info("No data available for detailed analysis.")
        return

    with st.expander("🔍 Detailed CSS Analysis", expanded=True):
        # 1. Select solutions within the SOI
        valid_ids = df["id"].dropna().astype(str).unique().tolist()
        selected_ids = st.multiselect(
            "Pick solutions to compare",
            options=valid_ids,
            default=valid_ids[:min(3, len(valid_ids))],
            help="Select at least 2 solutions for comparison."
        )

        if len(selected_ids) < 2:
            st.info("Select at least 2 solutions to enable detailed comparison.")
            return

        # Filter the DataFrame to the selected solutions
        compare_df = df[df["id"].astype(str).isin(selected_ids)].copy()

        # 2. Analysis tabs
        tab1, tab2, tab3 = st.tabs([
            "📊 Comparative Profile",
            "📋 Requirement Composition",
            "👥 Stakeholder Impact"
        ])

        # ----------------------------
        # Tab 1: Comparative Profile (Radar)
        # ----------------------------
        with tab1:
            st.subheader("Custom Trade-off Comparison")

            # Detect available numeric metrics (excluding id, req_*, and metadata)
            exclude_patterns = ("id", "req_", "x_", "var_", "item_", "selected_ids", "is_feasible", "scope")
            numeric_metrics = [
                c for c in compare_df.columns
                if pd.api.types.is_numeric_dtype(compare_df[c])
                and not any(c.startswith(p) for p in exclude_patterns)
                and c not in ["cluster", "cluster_str", "group_label", "group_base"]
            ]

            if len(numeric_metrics) < 3:
                st.warning("At least 3 numeric metrics are required for the radar chart.")
            else:
                selected_radar_metrics = st.multiselect(
                    "Select metrics (at least 3)",
                    options=numeric_metrics,
                    default=numeric_metrics[:3] if len(numeric_metrics) >= 3 else numeric_metrics,
                    key="radar_metrics"
                )

                if len(selected_radar_metrics) >= 3:
                    # Choose optimisation direction for each metric
                    metric_goals = {}
                    cols = st.columns(len(selected_radar_metrics))
                    for idx, m in enumerate(selected_radar_metrics):
                        with cols[idx]:
                            metric_goals[m] = st.selectbox(
                                f"Goal {m}",
                                ["Maximize", "Minimize"],
                                key=f"radar_goal_{m}"
                            )

                    # Normalise metrics to the range [0.1, 0.9] for the radar
                    radar_df = compare_df.copy()
                    low, high = 0.1, 0.9
                    for m in selected_radar_metrics:
                        mi, ma = radar_df[m].min(), radar_df[m].max()
                        if ma > mi:
                            norm = (radar_df[m] - mi) / (ma - mi)
                            if metric_goals[m] == "Minimize":
                                norm = 1.0 - norm
                            radar_df[m] = low + (norm * (high - low))
                        else:
                            radar_df[m] = 0.5

                    # Create the radar chart
                    fig_radar = go.Figure()
                    for _, row in radar_df.iterrows():
                        vals = row[selected_radar_metrics].tolist()
                        vals.append(vals[0])  # close the polygon
                        fig_radar.add_trace(go.Scatterpolar(
                            r=vals,
                            theta=selected_radar_metrics + [selected_radar_metrics[0]],
                            fill=None,
                            mode='lines+markers',
                            name=f"ID {row['id']}"
                        ))

                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=True,
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

        # ----------------------------
        # Tab 2: Requirement Composition
        # ----------------------------
        with tab2:
            st.subheader("Requirements Included in Selected Solutions")

            req_df = None

            # Case 1: CSV style (req_1, req_2, ...)
            req_cols = [c for c in compare_df.columns if c.startswith("req_")]

            if req_cols:
                req_df = compare_df.set_index("id")[req_cols].copy()

            # Case 2: Solutions with selected_ids column
            elif "selected_ids" in compare_df.columns:
                all_requirements = sorted({
                    req
                    for ids in compare_df["selected_ids"]
                    if isinstance(ids, (list, set, tuple))
                    for req in ids
                })

                if all_requirements:
                    matrix = []
                    for _, row in compare_df.iterrows():
                        selected = set(row["selected_ids"])
                        matrix.append({
                            req: 1 if req in selected else 0
                            for req in all_requirements
                        })
                    req_df = pd.DataFrame(
                        matrix,
                        index=[str(v) for v in compare_df["id"]]
                    )

            if req_df is None:
                st.info("No requirement information available.")
            else:
                req_df = req_df.fillna(0).astype(int)
                st.write("Shape:", req_df.shape)

                try:
                    fig_req = px.imshow(
                        req_df,
                        labels={
                            "x": "Requirements",
                            "y": "Solutions",
                            "color": "Status"
                        },
                        color_continuous_scale=[
                            [0, "#e0e0e0"],
                            [1, "#00e676"]
                        ]
                    )

                    fig_req.update_layout(
                        template="plotly_white",
                        coloraxis_showscale=False,
                        xaxis=dict(tickangle=-45),
                        yaxis=dict(autorange="reversed")
                    )

                    st.plotly_chart(fig_req, use_container_width=True)

                except Exception as e:
                    st.exception(e)

        # ----------------------------
        # Tab 3: Stakeholder Impact
        # ----------------------------
        with tab3:
            st.subheader("👥 Stakeholder Coverage & Alignment")

            # Obtain the plugin from the model
            plugin = None
            if model is not None and hasattr(model, "plugin"):
                plugin = model.plugin

            if plugin is None or not hasattr(plugin, "compute_stakeholder_coverage"):
                st.info("Stakeholder coverage analysis is only available for NRP domains with stakeholder data.")
            else:
                # Check for stakeholders in the model
                stakeholders = []
                if hasattr(model, "evaluators") and "stakeholders" in model.evaluators:
                    stakeholders = model.evaluators["stakeholders"]
                elif hasattr(model, "metadata") and "stakeholders" in model.metadata:
                    stakeholders = model.metadata["stakeholders"]

                if not stakeholders:
                    st.warning("No stakeholders found in the model. Make sure the JSON has a 'stakeholders' section.")
                else:
                    # Compute coverage
                    cov_df = plugin.compute_stakeholder_coverage(compare_df, model)

                    # Add coverage columns to compare_df
                    added_cols = 0
                    for col in cov_df.columns:
                        if col not in compare_df.columns:
                            compare_df[col] = cov_df[col]
                            added_cols += 1

                    if added_cols == 0:
                        st.warning("No coverage columns could be added. Verify that the model has original satisfaction data.")
                    else:
                        st.success(f"Added {added_cols} coverage column(s).")

                    # Identify coverage columns
                    cov_cols = [c for c in compare_df.columns if c.startswith("stcov_")]
                    if not cov_cols:
                        st.warning("No stakeholder coverage columns found. Radar cannot be displayed.")
                    else:
                        # Coverage radar
                        st.markdown("#### Coverage per Stakeholder (Radar)")

                        # Sort stakeholders by average coverage
                        cov_cols_sorted = sorted(
                            cov_cols,
                            key=lambda c: compare_df[c].mean(),
                            reverse=True
                        )

                        selected_st = st.multiselect(
                            "Select stakeholders to display",
                            cov_cols_sorted,
                            default=cov_cols_sorted[:min(6, len(cov_cols_sorted))],
                            key="st_radar_select"
                        )

                        if len(selected_st) < 3:
                            st.warning("Select at least 3 stakeholders for the radar chart.")
                        else:
                            # Normalise for the radar
                            radar_df = compare_df.copy()
                            low, high = 0.1, 0.9
                            for st_col in selected_st:
                                mi, ma = radar_df[st_col].min(), radar_df[st_col].max()
                                if ma > mi:
                                    norm = (radar_df[st_col] - mi) / (ma - mi)
                                    radar_df[st_col] = low + (norm * (high - low))
                                else:
                                    radar_df[st_col] = 0.5

                            fig_radar_st = go.Figure()
                            for _, row in radar_df.iterrows():
                                vals = row[selected_st].tolist()
                                vals.append(vals[0])
                                fig_radar_st.add_trace(go.Scatterpolar(
                                    r=vals,
                                    theta=selected_st + [selected_st[0]],
                                    fill=None,
                                    mode='lines+markers',
                                    name=f"ID {row['id']}"
                                ))

                            fig_radar_st.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                showlegend=True,
                                margin=dict(l=40, r=40, t=40, b=40)
                            )
                            st.plotly_chart(fig_radar_st, use_container_width=True)

                        # Alignment matrix
                        st.markdown("#### Alignment Matrix (Requirements vs Stakeholders)")

                        matrix = plugin.build_stakeholder_requirement_matrix(model) if plugin else None
                        if matrix is None:
                            st.info("No stakeholder-request matrix available. Alignment cannot be displayed.")
                        else:
                            valid_ids = compare_df["id"].dropna().astype(str).unique().tolist()
                            if not valid_ids:
                                st.warning("No valid solution IDs found.")
                            else:
                                focus_id = st.selectbox(
                                    "Select Solution to analyze alignment",
                                    valid_ids,
                                    key="align_solution_select"
                                )
                                row_focus = compare_df[compare_df["id"].astype(str) == focus_id].iloc[0]

                                req_cols = [c for c in compare_df.columns if c.startswith("req_")]
                                if not req_cols:
                                    st.info("No requirement columns found.")
                                else:
                                    st_ids = list(matrix.columns)
                                    alignment_data = []

                                    for st_id in st_ids:
                                        row_values = []
                                        for req in req_cols:
                                            req_id = req.replace("req_", "")
                                            proposed = matrix.loc[req_id, st_id] if req_id in matrix.index else 0
                                            included = row_focus.get(req, 0) == 1

                                            if proposed and included:
                                                val = 2
                                            elif proposed and not included:
                                                val = 1
                                            else:
                                                val = 0
                                            row_values.append(val)
                                        alignment_data.append(row_values)

                                    summary_row = [3 if row_focus.get(req, 0) == 1 else 1 for req in req_cols]
                                    alignment_data.append(summary_row)

                                    y_labels = [f"Stakeholder {st_id}" for st_id in st_ids] + ["📦 RELEASE STATUS"]

                                    align_df = pd.DataFrame(alignment_data, index=y_labels, columns=req_cols)

                                    fig_align = px.imshow(
                                        align_df,
                                        labels=dict(x="Requirements", y="", color="Status"),
                                        color_continuous_scale=[
                                            [0.0, "#f8f9fa"],
                                            [0.33, "#adb5bd"],
                                            [0.66, "#00e676"],
                                            [1.0, "#00695c"]
                                        ]
                                    )
                                    fig_align.update_layout(
                                        template="plotly_white",
                                        coloraxis_showscale=False,
                                        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
                                        yaxis=dict(tickfont=dict(size=11)),
                                        height=450
                                    )
                                    fig_align.update_traces(xgap=3, ygap=3)
                                    st.plotly_chart(fig_align, use_container_width=True)

                                    # Legend
                                    col1, col2, col3, col4 = st.columns(4)
                                    col1.markdown("⚪ **Not requested**")
                                    col2.markdown("🔘 **Requested (not included)**")
                                    col3.markdown("🟢 **Requested & Included**")
                                    col4.markdown("🌲 **Included in Release (summary)**")