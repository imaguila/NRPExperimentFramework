# Location: src/interface/controls/framing.py
"""
Framing Panel UI Component.

This module provides Streamlit view components for interactive bounding‑range
filters in the sidebar, allowing users to filter the decision space by
numeric dimensions via sliders.
"""

from typing import Dict, List, Tuple, Any, Optional
import streamlit as st
import pandas as pd


def render_framing_summary(total_count: int, remaining_count: int, container: Any = None) -> None:
    """
    Render a progress bar and ratio metrics summarising solution reduction.

    Args:
        total_count: Total number of solutions before framing.
        remaining_count: Number of solutions after applying current bounds.
        container: Streamlit container to render into (defaults to st).
    """
    if container is None:
        container = st

    ratio = remaining_count / max(total_count, 1)
    container.progress(ratio)

    container.markdown(
        f"""
        <div style="text-align:center; margin-bottom: 8px;">
            <div style="font-size:0.85rem;color:gray;">
                Visible Decision Space
            </div>
            <div style="font-size:1.4rem;font-weight:bold;">
                {remaining_count} / {total_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    container.caption(f"**{ratio:.1%}** of solutions satisfy framing bounds.")


def render_framing_panel(
    df: pd.DataFrame,
    total_count: Optional[int] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Render sliders for all available numeric dimensions in the DataFrame.

    Uses the minimum and maximum values from the current DataFrame.
    If all values for a dimension are equal, an artificial range is created
    so that the slider still appears.

    Args:
        df: The DataFrame containing the data to filter.
        total_count: Total number of solutions (optional; defaults to len(df)).

    Returns:
        A dictionary mapping dimension names to selected (min, max) bounds.
    """
    if df is None or df.empty:
        st.sidebar.info("ℹ️ Load a dataset to enable framing.")
        return {}

    # Include all numeric columns (including enriched ones) – no exclusion
    dimensions = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not str(c).startswith(("req_", "x_", "var_", "item_"))
        and str(c).lower() not in ("id", "selected_ids", "selected_ids_str", "is_feasible")
    ]

    if not dimensions:
        st.sidebar.caption("ℹ️ No numeric dimensions available for framing.")
        return {}

    active_bounds: Dict[str, Tuple[float, float]] = {}

    with st.sidebar.expander("🎬 Framing", expanded=False):
        # Render sliders for all dimensions
        for dim in dimensions:
            if dim in df.columns:
                s = pd.to_numeric(df[dim], errors="coerce").dropna()
                if s.empty:
                    continue

                min_v = float(s.min())
                max_v = float(s.max())

                # If all values are equal, create an artificial range
                if abs(max_v - min_v) < 1e-6:
                    if min_v == 0.0:
                        min_v_art = 0.0
                        max_v_art = 1.0
                    else:
                        min_v_art = max(0.0, min_v - 1.0)
                        max_v_art = min_v + 1.0

                    st.caption(f"**{dim.capitalize()}** (all values equal to {min_v:.2f})")
                    selected_range = st.slider(
                        f"{dim.capitalize()} (artificial range)",
                        min_value=min_v_art,
                        max_value=max_v_art,
                        value=(min_v_art, max_v_art),
                        step=0.01,
                        key=f"sidebar_framing_slider_{dim}"
                    )
                    active_bounds[dim] = selected_range
                    continue

                step_val = max((max_v - min_v) / 100.0, 0.01)

                selected_range = st.slider(
                    f"{dim.capitalize()}",
                    min_value=min_v,
                    max_value=max_v,
                    value=(min_v, max_v),
                    step=step_val,
                    key=f"sidebar_framing_slider_{dim}"
                )

                active_bounds[dim] = selected_range

        # Real-time preview of the filtered dataset
        temp_df = df.copy()
        for col, (min_b, max_b) in active_bounds.items():
            if col in temp_df.columns:
                s_col = pd.to_numeric(temp_df[col], errors="coerce")
                temp_df = temp_df[(s_col >= min_b) & (s_col <= max_b)]

        rem_cnt = len(temp_df)
        tot_cnt = total_count if total_count is not None else len(df)

        st.markdown("---")

        # Summary of remaining solutions inside the expander
        render_framing_summary(tot_cnt, rem_cnt, container=st)

    return active_bounds