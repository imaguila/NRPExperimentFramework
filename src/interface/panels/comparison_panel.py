# Location: src/interface/panels/comparison_panel.py

"""
Pareto Front Comparison Panel.

This module provides a UI panel for comparing multiple Pareto fronts
using a metrics table and a radar chart.

Only explicitly configured optimisation objectives are used to calculate
Pareto-front quality measures. Numeric descriptive attributes and derived
indicators are not interpreted as objectives.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.domain.core.paretofront import ParetoFront


VALID_DIRECTIONS = {"min", "max"}


def _normalise_direction(value: Any) -> str:
    """
    Convert a direction value or enum into 'min' or 'max'.
    """

    normalised = str(
        getattr(value, "value", value)
    ).strip().lower()

    aliases = {
        "minimise": "min",
        "minimize": "min",
        "minimisation": "min",
        "minimization": "min",
        "maximise": "max",
        "maximize": "max",
        "maximisation": "max",
        "maximization": "max",
    }

    normalised = aliases.get(
        normalised,
        normalised,
    )

    if normalised not in VALID_DIRECTIONS:
        raise ValueError(
            f"Invalid objective direction {value!r}. "
            "Expected 'min' or 'max'."
        )

    return normalised


def _normalise_directions(
    directions: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Normalise an objective-direction mapping.
    """

    if not isinstance(directions, dict):
        return {}

    result: Dict[str, str] = {}

    for name, direction in directions.items():
        result[str(name)] = _normalise_direction(
            direction
        )

    return result


def _resolve_objective_configuration(
    front: ParetoFront,
    problem=None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Resolve the objective names and directions for a Pareto front.

    Resolution order:
        1. Saved-front metadata.
        2. Associated OptimizationProblem.
        3. Directions stored in the ParetoFront.

    Directions are never inferred by assigning a default direction to
    every numeric column. Descriptive attributes such as dissatisfaction,
    risk, or enrichment indicators must not be treated as objectives
    unless they were explicitly included in the optimisation configuration.
    """

    metadata = metadata or {}

    # ---------------------------------------------------------
    # 1. Saved-front metadata
    # ---------------------------------------------------------

    metadata_directions = (
        metadata.get("objective_directions")
        or metadata.get("objectives_config")
        or metadata.get("objectives")
    )

    directions = _normalise_directions(
        metadata_directions
    )

    metadata_objective_columns = (
        metadata.get("objective_cols")
        or metadata.get("objective_names")
        or []
    )

    objective_names = [
        str(name)
        for name in metadata_objective_columns
        if str(name) in directions
    ]

    if directions:
        if not objective_names:
            objective_names = list(
                directions.keys()
            )

        return objective_names, directions

    # ---------------------------------------------------------
    # 2. Associated OptimizationProblem
    # ---------------------------------------------------------

    if (
        problem is not None
        and hasattr(
            problem,
            "get_objective_directions",
        )
    ):
        problem_directions = (
            problem.get_objective_directions()
        )

        directions = _normalise_directions(
            problem_directions
        )

        if directions:
            return (
                list(directions.keys()),
                directions,
            )

    if (
        problem is not None
        and hasattr(problem, "objectives")
    ):
        directions = _normalise_directions(
            getattr(problem, "objectives", {})
        )

        if directions:
            return (
                list(directions.keys()),
                directions,
            )

    # ---------------------------------------------------------
    # 3. Directions stored in the ParetoFront
    # ---------------------------------------------------------

    front_directions = _normalise_directions(
        getattr(
            front,
            "objective_directions",
            {},
        )
    )

    if front_directions:
        return (
            list(front_directions.keys()),
            front_directions,
        )

    # ---------------------------------------------------------
    # No safe configuration available
    # ---------------------------------------------------------

    raise ValueError(
        "Objective names and directions could not be resolved. "
        "The Pareto front comparison requires an explicit objective "
        "configuration. Numeric descriptive attributes cannot be "
        "assigned objective directions automatically."
    )


def _create_front_from_saved_data(
    saved_front: Dict[str, Any],
) -> ParetoFront:
    """
    Reconstruct a ParetoFront from a saved DataFrame.

    Only the explicitly saved objective columns are supplied as
    optimisation objectives.
    """

    objective_columns = list(
        saved_front.get(
            "objective_cols",
            [],
        )
        or []
    )

    saved_directions = _normalise_directions(
        saved_front.get(
            "objective_directions",
        )
        or saved_front.get(
            "objectives_config",
        )
        or saved_front.get(
            "objectives",
        )
    )

    if not objective_columns and saved_directions:
        objective_columns = list(
            saved_directions.keys()
        )

    if not objective_columns:
        raise ValueError(
            f"Saved front "
            f"{saved_front.get('name', '<unnamed>')!r} "
            "does not contain an explicit objective-column "
            "configuration."
        )

    front = ParetoFront.from_dataframe(
        df=saved_front["df"],
        decision_cols=saved_front.get(
            "decision_cols",
            [],
        ),
        objective_cols=objective_columns,
        objective_directions=(
            saved_directions
            if saved_directions
            else None
        ),
    )

    return front


def _compute_front_metrics(
    front: ParetoFront,
    problem=None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate metrics using only the explicitly configured objectives.
    """

    objective_names, directions = (
        _resolve_objective_configuration(
            front=front,
            problem=problem,
            metadata=metadata,
        )
    )

    missing_directions = [
        name
        for name in objective_names
        if name not in directions
    ]

    if missing_directions:
        raise ValueError(
            "Missing directions for configured objectives: "
            f"{missing_directions}."
        )

    front.set_objective_directions(
        directions
    )

    return front.compute_metrics(
        objective_names=objective_names,
        directions=directions,
    )


def _normalise_radar_column(
    series: pd.Series,
    lower_is_better: bool,
) -> pd.Series:
    """
    Normalise a metric column into [0, 1] for radar visualisation.

    Missing values remain missing and are not interpreted as zero.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    valid = numeric.dropna()

    if valid.empty:
        return result

    min_value = float(valid.min())
    max_value = float(valid.max())

    if abs(max_value - min_value) <= 1e-9:
        result.loc[valid.index] = 0.5
        return result

    if lower_is_better:
        result.loc[valid.index] = (
            max_value - valid
        ) / (
            max_value - min_value
        )
    else:
        result.loc[valid.index] = (
            valid - min_value
        ) / (
            max_value - min_value
        )

    return result


def render_comparison_panel(
    current_front: ParetoFront,
    problem,
) -> None:
    """
    Render the Pareto-front comparison panel.

    Saved fronts and the current front are compared only when their
    objective names and directions can be resolved explicitly.
    """

    saved_paretos = st.session_state.get(
        "saved_pareto_fronts",
        [],
    )

    if not saved_paretos:
        st.info(
            "No saved Pareto fronts to compare. "
            "Run optimisations and save them first."
        )
        return

    saved_names = [
        saved_front["name"]
        for saved_front in saved_paretos
    ]

    selected_names = st.multiselect(
        "Select Pareto Fronts to Compare",
        options=saved_names,
        default=saved_names[
            : min(3, len(saved_names))
        ],
    )

    if not selected_names:
        st.info(
            "Select at least one saved front "
            "to compare."
        )
        return

    comparison_entries: List[
        Tuple[
            str,
            ParetoFront,
            Optional[Dict[str, Any]],
        ]
    ] = []

    # ---------------------------------------------------------
    # Reconstruct selected saved fronts
    # ---------------------------------------------------------

    for selected_name in selected_names:
        saved_front = next(
            (
                value
                for value in saved_paretos
                if value["name"] == selected_name
            ),
            None,
        )

        if saved_front is None:
            continue

        try:
            reconstructed_front = (
                _create_front_from_saved_data(
                    saved_front
                )
            )

            comparison_entries.append(
                (
                    selected_name,
                    reconstructed_front,
                    saved_front,
                )
            )

        except (ValueError, KeyError) as error:
            st.warning(
                f"Front {selected_name!r} was not included: "
                f"{error}"
            )

    # ---------------------------------------------------------
    # Add current front
    # ---------------------------------------------------------

    if (
        current_front is not None
        and len(current_front) > 0
    ):
        comparison_entries.append(
            (
                "Current Front",
                current_front,
                None,
            )
        )

    if not comparison_entries:
        st.info(
            "No compatible fronts are available "
            "for comparison."
        )
        return

    # ---------------------------------------------------------
    # Verify objective compatibility
    # ---------------------------------------------------------

    compatible_entries = []
    reference_names = None
    reference_directions = None

    for (
        front_name,
        front,
        metadata,
    ) in comparison_entries:

        try:
            objective_names, directions = (
                _resolve_objective_configuration(
                    front=front,
                    problem=problem,
                    metadata=metadata,
                )
            )

        except ValueError as error:
            st.warning(
                f"Front {front_name!r} was not included: "
                f"{error}"
            )
            continue

        if reference_names is None:
            reference_names = objective_names
            reference_directions = directions

        if objective_names != reference_names:
            st.warning(
                f"Front {front_name!r} was not included because "
                f"its objectives {objective_names} differ from "
                f"the reference objectives {reference_names}."
            )
            continue

        if directions != reference_directions:
            st.warning(
                f"Front {front_name!r} was not included because "
                "its objective directions differ from the "
                "reference configuration."
            )
            continue

        compatible_entries.append(
            (
                front_name,
                front,
                metadata,
            )
        )

    if not compatible_entries:
        st.info(
            "No fronts have a compatible objective "
            "configuration."
        )
        return

    # ---------------------------------------------------------
    # Compute metrics
    # ---------------------------------------------------------

    metrics_data = []

    for (
        front_name,
        front,
        metadata,
    ) in compatible_entries:

        try:
            metrics = _compute_front_metrics(
                front=front,
                problem=problem,
                metadata=metadata,
            )

        except ValueError as error:
            st.warning(
                f"Metrics could not be calculated for "
                f"{front_name!r}: {error}"
            )
            continue

        metrics["Front Name"] = front_name
        metrics_data.append(metrics)

    if not metrics_data:
        st.info(
            "No comparable metric results were produced."
        )
        return

    df_metrics = pd.DataFrame(
        metrics_data
    )

    required_columns = [
        "Front Name",
        "Solutions",
        "Hypervolume",
        "Spacing",
        "Objective-space extent",
    ]

    available_columns = [
        column
        for column in required_columns
        if column in df_metrics.columns
    ]

    df_metrics = df_metrics[
        available_columns
    ]

    # ---------------------------------------------------------
    # Display comparison table
    # ---------------------------------------------------------

    st.dataframe(
        df_metrics,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="📥 Export Comparison Metrics (.csv)",
        data=df_metrics.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "pareto_comparison_metrics.csv"
        ),
        mime="text/csv",
        use_container_width=True,
        key="btn_export_comparison_metrics",
    )

    # ---------------------------------------------------------
    # Radar chart
    # ---------------------------------------------------------

    if len(df_metrics) <= 1:
        return

    radar_metrics = [
        column
        for column in df_metrics.columns
        if column != "Front Name"
        and pd.api.types.is_numeric_dtype(
            df_metrics[column]
        )
    ]

    # Metrics entirely composed of NaN cannot be plotted.
    radar_metrics = [
        column
        for column in radar_metrics
        if df_metrics[column].notna().any()
    ]

    if len(radar_metrics) < 3:
        st.info(
            "At least three valid metrics are required "
            "to display a radar chart."
        )
        return

    df_normalised = df_metrics.copy()

    for column in radar_metrics:
        df_normalised[column] = (
            _normalise_radar_column(
                df_metrics[column],
                lower_is_better=(
                    column == "Spacing"
                ),
            )
        )

    figure = go.Figure()

    for _, row in df_normalised.iterrows():

        values = []

        for column in radar_metrics:
            value = row[column]

            # Plotly cannot represent NaN reliably in a closed
            # radar polygon. Missing values remain visible as gaps.
            values.append(
                None
                if pd.isna(value)
                else float(value)
            )

        figure.add_trace(
            go.Scatterpolar(
                r=values,
                theta=radar_metrics,
                fill=None,
                mode="lines+markers",
                name=row["Front Name"],
                line=dict(width=2),
                marker=dict(size=6),
                connectgaps=False,
            )
        )

    figure.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[
                    0,
                    0.25,
                    0.5,
                    0.75,
                    1,
                ],
                ticktext=[
                    "0",
                    "0.25",
                    "0.5",
                    "0.75",
                    "1",
                ],
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05,
        ),
        margin=dict(
            l=40,
            r=100,
            t=40,
            b=40,
        ),
        height=500,
        template="plotly_white",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )