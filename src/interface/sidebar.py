# Location: src/interface/sidebar.py

"""
Sidebar Coordinator for Streamlit UI.

This module orchestrates all control panels in the sidebar, including
data loading, problem configuration, solver selection, and analysis
controls such as enrichment, framing, analytical lenses, and detailed
CSS analysis.

The execution of the optimisation algorithm is delegated to the
headless run_optimization application use case.
"""

from typing import Any, Dict, Optional, Tuple

import streamlit as st

from src.analysis.session import AnalysisSession
from src.application.use_cases.run_optimization import run_optimization
from src.domain.core.model import OptimizationModel
from src.domain.plugins.base_domain import DomainPlugin
from src.interface.controls.data_loader import render_data_loader_panel
from src.interface.controls.optimization_config import (
    render_optimization_config_panel,
)
from src.interface.controls.solver import render_solver_panel
from src.interface.session_state import reset_workspace_state
from src.solving.core.problem import OptimizationProblem
from src.solving.core.solver import BaseSolver


def render_sidebar(
    plugin: DomainPlugin,
) -> Tuple[
    Optional[OptimizationModel],
    Dict[str, Any],
    Dict[str, str],
    Optional[Any],
    float,
    Optional[BaseSolver],
    Optional[OptimizationProblem],
]:
    """
    Render the complete sidebar with all control panels.

    The sidebar is organised into two main blocks:

    A. Optimisation:
       Data loading, problem configuration, and solver execution.

    B. Analysis:
       Enrichment, framing, analytical lenses, and detailed CSS analysis.

    Args:
        plugin:
            Active domain plugin, for example NRPPlugin.

    Returns:
        A tuple containing:

        - model:
          Loaded or selected OptimizationModel, if available.

        - constraints_config:
          Dictionary of constraint specifications.

        - objectives_config:
          Dictionary of objective directions.

        - pareto_df:
          DataFrame of the current Pareto front, if available.

        - execution_time:
          Time taken for the last optimisation run.

        - solver:
          Solver instance used in the last execution, if available.

        - problem:
          OptimizationProblem from the last execution, if available.
    """
    with st.sidebar:

        # ============================================================
        # RESET ENVIRONMENT
        # ============================================================

        if st.button(
            "🔄 Reset Environment",
            key="reset_button",
            use_container_width=True,
        ):
            reset_workspace_state()
            st.rerun()

        st.title(
            f"⚡ {plugin.get_display_name()}"
        )

        st.markdown(
            "## :blue[Data Acquisition & Provisioning]"
        )

        # ============================================================
        # DATA LOADING
        # ============================================================

        model, _raw_data = render_data_loader_panel(
            plugin=plugin,
        )

        if model is not None:
            st.session_state["last_model"] = model

        # ============================================================
        # CURRENT OPTIMISATION RESULT
        # ============================================================

        pareto_df = st.session_state.get(
            "last_pareto_df"
        )

        execution_time = st.session_state.get(
            "last_execution_time",
            0.0,
        )

        solver = st.session_state.get(
            "last_solver"
        )

        problem = st.session_state.get(
            "last_problem"
        )

        constraints_config: Dict[str, Any] = {}
        objectives_config: Dict[str, str] = {}

        # ============================================================
        # BLOCK A: PROBLEM FORMULATION AND OPTIMISATION
        # ============================================================

        if model is not None:
            st.markdown(
                "## :blue[Problem Formulation & Optimization]"
            )

            with st.expander(
                "🎯 Problem Configuration",
                expanded=True,
            ):
                (
                    constraints_config,
                    objectives_config,
                ) = render_optimization_config_panel(
                    model,
                    plugin=plugin,
                )

            if objectives_config:
                with st.expander(
                    "🚀 Solver Selection & Tuning",
                    expanded=True,
                ):
                    solver_result = render_solver_panel(
                        selected_model=model,
                        objectives_config=objectives_config,
                        constraints_config=constraints_config,
                        plugin=plugin,
                    )

                    if solver_result:
                        solver, problem = solver_result

                        # --------------------------------------------
                        # Execute the solver through the headless
                        # application use case.
                        # --------------------------------------------

                        with st.spinner(
                            "🚀 Running optimization..."
                        ):
                            run_result = run_optimization(
                                problem=problem,
                                solver=solver,
                            )

                        pareto_front = (
                            run_result.pareto_front
                        )

                        execution_time = (
                            run_result.execution_time
                        )

                        # Preserve objective-direction metadata exactly
                        # as in the previous sidebar implementation.
                        pareto_front.set_objective_directions(
                            problem.get_objective_directions()
                        )

                        # Rebuild the DataFrame after setting the
                        # objective-direction metadata.
                        pareto_df = (
                            pareto_front.to_dataframe()
                        )

                        st.success(
                            "✅ Pareto front generated with "
                            f"{len(pareto_df)} solutions in "
                            f"{execution_time:.2f}s"
                        )

                        # --------------------------------------------
                        # Store the current optimisation result
                        # --------------------------------------------

                        st.session_state[
                            "last_pareto_front"
                        ] = pareto_front

                        st.session_state[
                            "last_pareto_df"
                        ] = pareto_df

                        # Keep the current-result keys synchronized.
                        # These keys are also used by imported CSV fronts.
                        st.session_state[
                            "pareto_front"
                        ] = pareto_front

                        st.session_state[
                            "pareto_df"
                        ] = pareto_df

                        st.session_state[
                            "last_execution_time"
                        ] = execution_time

                        st.session_state[
                            "last_solver"
                        ] = run_result.solver

                        st.session_state[
                            "last_problem"
                        ] = run_result.problem

                        st.session_state[
                            "source_mode"
                        ] = "calculated_pareto"

                        # --------------------------------------------
                        # Reset analytical results derived from the
                        # previous Pareto front
                        # --------------------------------------------

                        st.session_state[
                            "analysis_df"
                        ] = pareto_df.copy()

                        st.session_state[
                            "analysis_df_after_lens"
                        ] = None

                        st.session_state[
                            "enriched_df"
                        ] = None

                        st.session_state[
                            "calculated_indicators"
                        ] = []

                        st.session_state[
                            "selected_indicators_multiselect"
                        ] = []

                        st.session_state[
                            "enrichment_just_calculated"
                        ] = False

                        st.session_state[
                            "saved_sois"
                        ] = []

                        st.session_state[
                            "active_soi_id"
                        ] = None

                        # Preserve Pareto fronts saved from previous solver executions.
                        # They are required for cross-solver comparison.
                        if "saved_pareto_fronts" not in st.session_state:
                            st.session_state[
                                "saved_pareto_fronts"
                            ] = []

                        st.session_state[
                            "maps"
                        ] = []
                        # --------------------------------------------
                        # Create a new analysis session
                        # --------------------------------------------

                        st.session_state[
                            "analysis_session"
                        ] = AnalysisSession(
                            base_front=pareto_front,
                            name=f"Analysis - {model.name}",
                            model=model,
                        )

                        st.info(
                            "✅ Analysis session ready. "
                            "Use the workspace to explore solutions."
                        )

        # ============================================================
        # BLOCK B: RESULTS AND DECISION ANALYSIS
        # ============================================================

        # Both calculated and imported fronts use "pareto_df" as
        # the current result key.
        current_pareto_df = st.session_state.get(
            "pareto_df"
        )

        if current_pareto_df is not None:
            pareto_df = current_pareto_df

        if pareto_df is not None and not pareto_df.empty:
            st.markdown(
                "## :blue[Results & Decision Analysis]"
            )

            # --------------------------------------------------------
            # Analysis mode
            # --------------------------------------------------------

            if (
                "analysis_mode_enabled"
                not in st.session_state
            ):
                st.session_state[
                    "analysis_mode_enabled"
                ] = False

            analysis_enabled = st.checkbox(
                "⚙️ Enable Analysis Mode",
                key="analysis_mode_enabled",
                help=(
                    "Enable this to show enrichment, framing, "
                    "lenses, and decision maps."
                ),
            )

            # --------------------------------------------------------
            # Detailed CSS analysis
            # --------------------------------------------------------

            if (
                "detailed_analysis_enabled"
                not in st.session_state
            ):
                st.session_state[
                    "detailed_analysis_enabled"
                ] = False

            st.checkbox(
                "🔍 Open Detailed Comparison (CSS Analysis)",
                key="detailed_analysis_enabled",
                help=(
                    "Enables an expander in the workspace for "
                    "in-depth solution comparison."
                ),
            )

            # Start from the original Pareto front.
            working_base_df = pareto_df.copy()

            if analysis_enabled:

                # ====================================================
                # ENRICHMENT
                # ====================================================

                from src.interface.controls.enrichment_panel import (
                    render_enrichment_panel,
                )

                render_enrichment_panel(
                    pareto_df,
                    model=model,
                )

                enriched_df = st.session_state.get(
                    "enriched_df"
                )

                if enriched_df is not None:
                    working_base_df = enriched_df.copy()

                # ====================================================
                # FRAMING
                # ====================================================

                from src.interface.controls.framing import (
                    render_framing_panel,
                )

                framing_bounds = render_framing_panel(
                    working_base_df,
                    total_count=len(working_base_df),
                )

                if framing_bounds:
                    from src.analysis.framing import (
                        apply_framing_bounds,
                    )

                    working_base_df = apply_framing_bounds(
                        working_base_df,
                        framing_bounds,
                    )

                # ====================================================
                # ANALYTICAL LENSES
                # ====================================================

                from src.interface.controls.lenses import (
                    render_lenses_panel,
                )

                (
                    analysis_df_from_lens,
                    _active_lens_name,
                    _lens_params,
                ) = render_lenses_panel(
                    df=working_base_df,
                    available_dimensions=list(
                        working_base_df.columns
                    ),
                )

                # ----------------------------------------------------
                # Resolve the final working DataFrame
                # ----------------------------------------------------

                analysis_df_after_lens = (
                    st.session_state.get(
                        "analysis_df_after_lens"
                    )
                )

                if analysis_df_after_lens is not None:
                    analysis_df = (
                        analysis_df_after_lens.copy()
                    )

                elif analysis_df_from_lens is not None:
                    analysis_df = (
                        analysis_df_from_lens.copy()
                    )

                else:
                    analysis_df = (
                        working_base_df.copy()
                    )

            else:
                # Analysis mode disabled: use the original Pareto.
                analysis_df = pareto_df.copy()

            # DataFrame read by app.py and workspace panels.
            st.session_state[
                "analysis_df"
            ] = analysis_df

    return (
        model,
        constraints_config,
        objectives_config,
        pareto_df,
        execution_time,
        solver,
        problem,
    )