# Location: src/analysis/enrichment.py
"""
Enrichment utilities for computing domain‑specific indicators.

This module provides functions to retrieve available enrichment indicators
and compute them on a DataFrame of solutions. It delegates to the model's
plugin when available, falling back to the NRPEnricher for backward
compatibility when no plugin is present (e.g., when working with loaded CSVs).
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def get_available_indicators(df: Optional[pd.DataFrame], model: Any = None) -> Dict[str, Dict[str, Any]]:
    """
    Inspect the DataFrame and return enrichment indicators that can be computed.

    If a model with a plugin is provided, the plugin's enrichment indicators
    are used. Otherwise, a fallback to NRPEnricher is attempted (for CSVs
    without a model).

    Args:
        df: DataFrame containing solutions.
        model: Optional OptimizationModel instance (may contain a plugin).

    Returns:
        A dictionary mapping indicator names to their metadata
        (description and required attributes).
    """
    if df is None or df.empty:
        return {}

    # 1. Try to obtain indicators from the model's plugin
    plugin = None
    if model is not None and hasattr(model, "plugin"):
        plugin = model.plugin

    if plugin is not None and hasattr(plugin, "get_enrichment_indicators"):
        all_indicators = plugin.get_enrichment_indicators()
        df_cols_lower = {str(c).lower() for c in df.columns}
        available = {}
        for name, info in all_indicators.items():
            required = info.get("required", [])
            if not required or all(req.lower() in df_cols_lower for req in required):
                available[name] = info
        return available

    # 2. Fallback: use NRPEnricher directly (for CSV‑loaded data)
    try:
        from src.domain.plugins.nrp_enrichment import NRPEnricher
        indicators = NRPEnricher.get_calculated_indicators()
        df_cols_lower = {str(c).lower() for c in df.columns}
        available = {}
        for name, definition in indicators.items():
            reqs = definition.required_attributes
            if not reqs or all(req.lower() in df_cols_lower for req in reqs):
                available[name] = {
                    "description": definition.description,
                    "required": reqs
                }
        return available
    except ImportError:
        logger.warning("NRPEnricher not available for fallback.")
        return {}


def compute_enrichment_indicators(
    df: pd.DataFrame,
    selected_indicators: List[str],
    model: Any = None
) -> pd.DataFrame:
    """
    Compute the requested enrichment indicators on the DataFrame.

    The computation is delegated to the model's plugin if available,
    otherwise falls back to NRPEnricher.

    Args:
        df: Input DataFrame containing solutions.
        selected_indicators: List of indicator names to compute.
        model: Optional OptimizationModel instance (used to obtain the plugin).

    Returns:
        A new DataFrame with the computed indicators added as columns.
        If no indicators are computed, the original DataFrame is returned.
    """
    if df is None or df.empty or not selected_indicators:
        return df

    # 1. Try to use the plugin
    plugin = None
    if model is not None and hasattr(model, "plugin"):
        plugin = model.plugin

    if plugin is not None and hasattr(plugin, "compute_enrichment"):
        # Detect decision variable prefix
        decision_prefix = "req_"
        for prefix in ["req_", "item_", "x_", "var_"]:
            if any(str(c).startswith(prefix) for c in df.columns):
                decision_prefix = prefix
                break

        try:
            return plugin.compute_enrichment(
                df=df,
                indicators=selected_indicators,
                decision_var_prefix=decision_prefix
            )
        except Exception as exc:
            logger.error("Error computing indicators with plugin: %s", exc)
            # Continue to fallback

    # 2. Fallback: use NRPEnricher
    try:
        from src.domain.plugins.nrp_enrichment import NRPEnricher
        # Detect decision prefix
        decision_prefix = "req_"
        for prefix in ["req_", "item_", "x_", "var_"]:
            if any(str(c).startswith(prefix) for c in df.columns):
                decision_prefix = prefix
                break
        return NRPEnricher.compute_indicators(df, selected_indicators, decision_prefix)
    except ImportError:
        logger.error("NRPEnricher not available for fallback.")
        return df