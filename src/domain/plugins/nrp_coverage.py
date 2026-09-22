# Location: src/domain/plugins/nrp_coverage.py
"""
NRP Stakeholder Coverage utilities.

This module provides functions to compute stakeholder coverage for solutions
in the Next Release Problem (NRP) domain. It extracts original satisfaction
values from the raw data, builds a stakeholder-requirement request matrix,
and computes coverage scores for each solution.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)


def _get_original_satisfaction(model) -> Dict[str, List[float]]:
    """
    Retrieve the original (non‑aggregated) satisfaction values from the model.

    First checks `model.metadata["original_satisfaction"]` for cached values.
    If not present, it builds the mapping from the model's raw_data or from
    the items themselves, preserving the multivalued lists.

    Args:
        model: The OptimizationModel instance.

    Returns:
        Dictionary mapping requirement IDs to lists of satisfaction values
        (one per stakeholder, in order).
    """
    if model.metadata and "original_satisfaction" in model.metadata:
        return model.metadata["original_satisfaction"]

    original = {}
    # Try to read from raw_data (the original JSON)
    if model.raw_data and "requirements" in model.raw_data:
        for req in model.raw_data["requirements"]:
            req_id = str(req.get("id"))
            attrs = req.get("attributes", {})
            sat_vals = attrs.get("satisfaction", [])
            if isinstance(sat_vals, list):
                original[req_id] = [float(v) for v in sat_vals]
            else:
                # Fallback: try to get from the item itself (may already be aggregated)
                item = model.items.get(req_id)
                if item and hasattr(item, "attributes"):
                    sat = item.attributes.get("satisfaction")
                    if isinstance(sat, list):
                        original[req_id] = [float(v) for v in sat]
                    else:
                        original[req_id] = []
                else:
                    original[req_id] = []
    else:
        # Fallback: iterate over items directly
        for req_id, item in model.items.items():
            sat = item.attributes.get("satisfaction")
            if isinstance(sat, list):
                original[req_id] = [float(v) for v in sat]
            else:
                original[req_id] = []

    # Cache for future calls
    if model.metadata is None:
        model.metadata = {}
    model.metadata["original_satisfaction"] = original
    return original


def build_stakeholder_requirement_matrix(model) -> Optional[pd.DataFrame]:
    """
    Build the stakeholder‑requirement request matrix from the model.

    The matrix has requirements as rows and stakeholders as columns.
    A value of 1 indicates that the stakeholder requested the requirement
    (satisfaction > 0), and 0 otherwise.

    Args:
        model: The OptimizationModel instance.

    Returns:
        A DataFrame with requirements as index and stakeholder IDs as columns,
        or None if no stakeholders are found or data is missing.
    """
    if not model or not model.items:
        logger.warning("Model has no items. Cannot build stakeholder matrix.")
        return None

    # Retrieve stakeholders from evaluators or metadata
    stakeholders = []
    if hasattr(model, "evaluators") and "stakeholders" in model.evaluators:
        stakeholders = model.evaluators["stakeholders"]
    elif hasattr(model, "metadata") and "stakeholders" in model.metadata:
        stakeholders = model.metadata["stakeholders"]

    if not stakeholders:
        logger.warning("No stakeholders found in the model.")
        return None

    st_ids = [str(s.id) if hasattr(s, "id") else str(s.get("id")) for s in stakeholders]
    num_st = len(st_ids)

    sat_by_req = _get_original_satisfaction(model)
    if not sat_by_req:
        logger.warning("No original satisfaction values retrieved.")

    data = {}
    for req_id in model.items.keys():
        sat_vals = sat_by_req.get(req_id, [])
        # Ensure the list has the same length as the number of stakeholders
        if len(sat_vals) != num_st:
            sat_vals = sat_vals[:num_st] + [0] * (num_st - len(sat_vals))
        # Convert to binary: 1 if satisfaction > 0 else 0
        req_vector = [1 if v > 0 else 0 for v in sat_vals]
        data[req_id] = req_vector

    if not data:
        logger.warning("Could not build matrix data.")
        return None

    df_matrix = pd.DataFrame.from_dict(data, orient="index", columns=st_ids)
    if model.metadata is None:
        model.metadata = {}
    model.metadata["stakeholder_matrix"] = df_matrix
    return df_matrix


def compute_stakeholder_coverage(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Compute stakeholder coverage for each solution in the DataFrame.

    For each solution (row), the function identifies which requirements are
    selected (based on `req_*` columns) and sums the satisfaction values of
    those requirements for each stakeholder. The resulting columns are added
    as `stcov_<stakeholder_id>`.

    If no `req_*` columns are found, the function attempts to reconstruct them
    from the `selected_ids` column (if present).

    Args:
        df: DataFrame containing solutions (rows).
        model: The OptimizationModel instance.

    Returns:
        The original DataFrame with additional `stcov_*` columns for each stakeholder.
    """
    if df is None or df.empty or model is None:
        logger.warning("compute_stakeholder_coverage: insufficient data.")
        return df

    # 1. Ensure we have req_* columns
    req_cols = [c for c in df.columns if c.startswith("req_")]
    if not req_cols and "selected_ids" in df.columns:
        logger.info("No req_* columns found. Reconstructing from selected_ids...")
        all_req_ids = list(model.items.keys())
        for req_id in all_req_ids:
            col_name = f"req_{req_id}"
            df[col_name] = df["selected_ids"].apply(
                lambda ids: 1 if req_id in ids else 0 if isinstance(ids, (list, set, tuple)) else 0
            )
        req_cols = [c for c in df.columns if c.startswith("req_")]
        logger.info(f"Created {len(req_cols)} req_* columns.")

    if not req_cols:
        logger.warning("compute_stakeholder_coverage: no req_* or selected_ids columns found.")
        return df

    # 2. Build the stakeholder‑requirement matrix
    matrix = build_stakeholder_requirement_matrix(model)
    if matrix is None:
        logger.warning("compute_stakeholder_coverage: could not obtain request matrix.")
        return df

    # 3. Get original satisfaction values per requirement
    sat_by_req = _get_original_satisfaction(model)
    if not sat_by_req:
        logger.warning("compute_stakeholder_coverage: no original satisfaction values.")
        return df

    # 4. Retrieve stakeholder IDs (from raw_data, evaluators, or metadata)
    stakeholders = []
    if model.raw_data and "stakeholders" in model.raw_data:
        stakeholders = model.raw_data["stakeholders"]
    elif hasattr(model, "evaluators") and "stakeholders" in model.evaluators:
        stakeholders = model.evaluators["stakeholders"]
    elif hasattr(model, "metadata") and "stakeholders" in model.metadata:
        stakeholders = model.metadata["stakeholders"]

    st_ids = [str(s.get("id")) if isinstance(s, dict) else str(s.id) for s in stakeholders]
    if not st_ids:
        logger.warning("compute_stakeholder_coverage: no stakeholder IDs found.")
        return df

    # 5. Compute coverage for each solution and stakeholder
    result = df.copy()
    for st_idx, st_id in enumerate(st_ids):
        st_col = f"stcov_{st_id}"
        coverage = []
        for _, row in result.iterrows():
            included_reqs = [req for req in req_cols if row.get(req, 0) == 1]
            total = 0.0
            for req in included_reqs:
                req_id = req.replace("req_", "")
                sat_vals = sat_by_req.get(req_id, [])
                if st_idx < len(sat_vals):
                    total += sat_vals[st_idx]
            coverage.append(total)
        result[st_col] = coverage

    logger.info(f"Added stakeholder coverage columns: {[c for c in result.columns if c.startswith('stcov_')]}")
    return result