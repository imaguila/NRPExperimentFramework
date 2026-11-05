"""
Visualization Module.

Contains:
- Utility functions (format_ids_compact)
- Summary metrics & table (render_summary_metrics, render_dataset_table, render_export_section)
- Plotly figure generation functions (scatter, coordinated maps, distribution, parallel coordinates)
- Color and hover helpers
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import streamlit as st


# =====================================================
# FORMAT UTILITIES
# =====================================================

def format_ids_compact(ids: Any) -> str:
    """Formats a list, string or sequence of IDs into a clean compact bracket notation."""
    if not ids:
        return "[]"
    if isinstance(ids, str):
        clean = [x.strip() for x in ids.split(",") if x.strip()]
    elif isinstance(ids, (list, tuple, set)):
        clean = [str(x).strip() for x in ids if str(x).strip()]
    else:
        clean = [str(ids).strip()]

    return f"[{', '.join(sorted(set(clean)))}]" if clean else "[]"


# =====================================================
# SUMMARY METRICS & TABLE
# =====================================================

def render_summary_metrics(
    df: pd.DataFrame,
    objectives_config: Optional[Dict[str, str]] = None,
    selected_model: Any = None,
) -> None:
    """Renders the 5 core dataset dimension metrics (Overview View)."""
    if df is None or df.empty:
        return

    if selected_model is not None and hasattr(selected_model, "items"):
        num_decision_vars = len(selected_model.items)
    else:
        var_cols = {col for col in df.columns if str(col).startswith(("x_", "var_", "req_", "item_"))}
        num_decision_vars = len(var_cols)

    if objectives_config:
        obj_cols = set(objectives_config.keys())
    else:
        obj_cols = {"cost", "score", "scope"}

    id_cols = {col for col in df.columns if str(col).lower() in ("id", "solution_id", "selected_ids")}

    all_decision_cols = {col for col in df.columns if str(col).startswith(("x_", "var_", "req_", "item_"))}
    enriched_cols = set(df.columns) - all_decision_cols - obj_cols - id_cols

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Active Solutions", len(df))
    with c2:
        st.metric("Decision Vars", num_decision_vars)
    with c3:
        st.metric("Objectives", len(obj_cols))
    with c4:
        st.metric("Enriched Attribs", len(enriched_cols))
    with c5:
        st.metric("Total Attribs", len(df.columns))


def render_dataset_table(df: pd.DataFrame, objectives_config: Optional[Dict[str, str]] = None) -> None:
    if df is None or df.empty:
        st.info("No data available to display.")
        return

    st.markdown("#### 📋 Current Dataset")
    display_df = df.copy()
    if "selected_ids" in display_df.columns:
        display_df["Selected Items"] = [
            format_ids_compact(row.get("selected_ids", [])) for _, row in display_df.iterrows()
        ]
    hidden_cols = ["selected_ids", "selected_ids_str", "is_feasible"]
    cols_to_show = [c for c in display_df.columns if c not in hidden_cols]
    st.dataframe(display_df[cols_to_show], use_container_width=True, hide_index=True)


def render_export_section(df: pd.DataFrame, key_prefix: str) -> None:
    st.download_button(
        label="📊 Export Current Set (.csv)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="current_set.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{key_prefix}_btn_export_summary_csv",
    )


# =====================================================
# COLOR & HOVER UTILITIES
# =====================================================

def infer_lens_color_column(df: pd.DataFrame, user_color: Optional[str] = None) -> Optional[str]:
    if "group_label" in df.columns:
        return "group_label"
    if "cluster_str" in df.columns:
        return "cluster_str"
    if "preference_score" in df.columns:
        return "preference_score"
    if "efficiency_score" in df.columns:
        return "efficiency_score"
    if "consensus_score" in df.columns:
        return "consensus_score"
    if "domain_match_count" in df.columns:
        return "domain_match_count"
    return user_color


def is_discrete_color(df: pd.DataFrame, color_column: Optional[str]) -> bool:
    if color_column is None or color_column not in df.columns:
        return False
    if color_column in ["group_label", "cluster_str", "preference_method", "efficiency_method", "domain_matched_metrics"]:
        return True
    return pd.api.types.is_object_dtype(df[color_column])


def build_hover_columns(df: pd.DataFrame) -> List[str]:
    excluded_prefixes = ("req_", "var_", "x_")
    excluded_cols = {"label", "highlight", "highlight_label"}
    return [col for col in df.columns if col not in excluded_cols and not col.startswith(excluded_prefixes)]


# =====================================================
# PLOTLY FIGURE BUILDERS
# =====================================================

def render_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    size: Optional[str] = None,
    color: Optional[str] = None,
    show_ids: bool = False,
    key: Optional[str] = None,
) -> None:
    df_plot = df.copy()
    if x not in df_plot.columns or y not in df_plot.columns:
        st.warning("Selected axes are not available in the current dataset.")
        return

    text_column = None
    if show_ids:
        if "id" in df_plot.columns:
            text_column = "id"
        elif "ID" in df_plot.columns:
            text_column = "ID"

    plot_color = infer_lens_color_column(df_plot, user_color=color)
    discrete_color = is_discrete_color(df_plot, plot_color)
    hover_cols = build_hover_columns(df_plot)

    if discrete_color and plot_color is not None:
        df_plot[plot_color] = df_plot[plot_color].astype(str)

    fig = px.scatter(
        df_plot,
        x=x,
        y=y,
        size=size if size in df_plot.columns else None,
        color=plot_color if plot_color in df_plot.columns else None,
        text=text_column,
        hover_data=hover_cols,
        template="plotly_white",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: Optional[str] = None,
    size: Optional[str] = None,
    color: Optional[str] = None,
    show_ids: bool = False,
    use_3d: bool = False,
    key: Optional[str] = None,
    height: int = 500,
    **kwargs
) -> None:
    """
    Render a scatter plot (2D, 3D, or bubble chart) with common options.

    This is a more comprehensive version than `render_scatter`, supporting 3D plots.

    Args:
        df: DataFrame containing the data.
        x, y, z: Column names for axes.
        size: Column for bubble size (optional).
        color: Column for color encoding (optional).
        show_ids: If True, display ID labels.
        use_3d: If True and z is not None, generate a 3D plot.
        key: Plotly chart key.
        height: Chart height in pixels.
        **kwargs: Additional arguments passed to px.scatter or px.scatter_3d.
    """
    if x not in df.columns or y not in df.columns:
        st.warning("Selected axes are not available.")
        return

    # Infer color if not provided
    if color is None:
        color = infer_lens_color_column(df)

    # Prepare hover data and text
    hover_cols = build_hover_columns(df)
    text_col = "id" if show_ids and "id" in df.columns else None

    # Base configuration
    base_kwargs = {
        "data_frame": df,
        "x": x,
        "y": y,
        "color": color if color and color in df.columns else None,
        "size": size if size and size in df.columns else None,
        "text": text_col,
        "hover_data": hover_cols,
        "template": "plotly_white",
    }
    base_kwargs.update(kwargs)

    if use_3d and z is not None and z in df.columns:
        fig = px.scatter_3d(**base_kwargs, z=z)
        fig.update_traces(marker=dict(size=6, opacity=0.85))
    else:
        fig = px.scatter(**base_kwargs)
        fig.update_traces(marker=dict(size=8, opacity=0.85))

    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=height)
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_coordinated_maps(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    key_prefix: str,
    show_ids: bool = False,
    color: Optional[str] = None,
    size: Optional[str] = None,
) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"**Map A:** {x} vs {y}")
        render_scatter(df, x=x, y=y, color=color, size=size, show_ids=show_ids, key=f"{key_prefix}_a")
    with col2:
        st.caption(f"**Map B:** {x} vs {z}")
        render_scatter(df, x=x, y=z, color=color, size=size, show_ids=show_ids, key=f"{key_prefix}_b")


def render_distribution(
    df: pd.DataFrame,
    metric: str,
    mode: str = "Violin",
    color: Optional[str] = None,
    key: Optional[str] = None,
) -> None:
    """
    Renders Violin or Box plot.
    - If `color` is provided, the plot is colored by that column.
    - Otherwise, a single color is used (default).
    """
    if metric not in df.columns:
        st.warning(f"Metric '{metric}' not found in dataframe.")
        return

    if color is None:
        color = infer_lens_color_column(df)

    if mode == "Violin":
        fig = px.violin(df, y=metric, color=color, box=True, points="all", template="plotly_white")
    else:
        fig = px.box(df, y=metric, color=color, points="all", template="plotly_white")

    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_parallel_coordinates(
    df: pd.DataFrame,
    dimensions: List[str],
    key: Optional[str] = None,
) -> None:
    if len(dimensions) < 2:
        st.info("Select at least 2 dimensions.")
        return

    color_col = infer_lens_color_column(df)
    fig = px.parallel_coordinates(
        df,
        dimensions=dimensions,
        color=color_col,
        template="plotly_white",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=500)
    st.plotly_chart(fig, use_container_width=True, key=key)