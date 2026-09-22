# Location: src/analysis/framing.py
"""
Framing utilities for filtering solution sets by numeric bounds.

This module provides functions to identify numeric dimensions suitable for
framing (filtering), compute their min/max bounds, and apply those bounds
to filter a DataFrame of solutions.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd


def get_framing_dimensions(
    metrics: Optional[List[str]] = None,
    indicators: Optional[List[str]] = None,
    df: Optional[pd.DataFrame] = None
) -> List[str]:
    """
    Combine metrics and indicators into a list of framing dimensions.

    If no explicit lists are provided, the function automatically infers
    numeric columns from the DataFrame, excluding decision variable columns
    (starting with 'req_', 'x_', 'var_', 'item_') and metadata columns
    ('id', 'selected_ids', 'selected_ids_str', 'is_feasible').

    Args:
        metrics: Optional list of metric column names.
        indicators: Optional list of indicator column names.
        df: Optional DataFrame for automatic inference (used if no metrics
            or indicators are provided).

    Returns:
        A deduplicated list of dimension names suitable for framing.
    """
    combined = []
    if metrics:
        combined.extend(metrics)
    if indicators:
        combined.extend(indicators)

    # Automatic inference if no explicit lists are provided
    if not combined and df is not None and not df.empty:
        excluded = {"id", "selected_ids", "selected_ids_str", "is_feasible"}
        combined = [
            c for c in df.columns
            if str(c).lower() not in excluded
            and not str(c).startswith(("req_", "x_", "var_", "item_"))
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    return list(dict.fromkeys(combined))


def get_dimension_bounds(
    df: pd.DataFrame, dimensions: List[str]
) -> Dict[str, Tuple[float, float]]:
    """
    Compute the min/max bounds for each numeric dimension.

    Args:
        df: DataFrame containing the data.
        dimensions: List of column names to compute bounds for.

    Returns:
        A dictionary mapping dimension names to (min, max) tuples.
        Only dimensions that are numeric and present in the DataFrame are included.
    """
    bounds: Dict[str, Tuple[float, float]] = {}
    if df is None or df.empty or not dimensions:
        return bounds

    for dim in dimensions:
        if dim in df.columns and pd.api.types.is_numeric_dtype(df[dim]):
            min_v = float(df[dim].min())
            max_v = float(df[dim].max())

            if not (pd.isna(min_v) or pd.isna(max_v)):
                bounds[dim] = (min_v, max_v)

    return bounds


def apply_framing_bounds(
    df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """
    Filter a DataFrame to keep only rows within the specified bounds.

    For each dimension, rows with values outside the [min, max] range are removed.
    If a dimension is not present in the DataFrame, it is ignored.

    Args:
        df: Input DataFrame.
        bounds: Dictionary mapping dimension names to (min, max) tuples.

    Returns:
        A filtered DataFrame with only rows satisfying all bounds.
        If no bounds are provided, the original DataFrame is returned.
    """
    if df is None or df.empty or not bounds:
        return df

    filtered_df = df.copy()
    for dim, (min_v, max_v) in bounds.items():
        if dim in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df[dim] >= min_v) & (filtered_df[dim] <= max_v)
            ]

    return filtered_df