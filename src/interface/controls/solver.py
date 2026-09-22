# Location: src/interface/controls/solver.py
"""
Solver Selection and Configuration Panel.

This module provides the UI panel for selecting an optimisation solver,
configuring its hyperparameters, running the optimisation, and saving
the resulting Pareto front.
"""

from typing import Any, Dict, Optional, Tuple
import streamlit as st
import pandas as pd

from src.domain.core.model import OptimizationModel
from src.domain.core.paretofront import ParetoFront
from src.domain.plugins.base_domain import DomainPlugin
from src.solving.core.problem import OptimizationProblem, ConstraintSpec
from src.solving.core.registry import get_registered_solvers, get_solver_class
from src.interface.components.widget_factory import render_parameter_widget


def render_solver_panel(
    selected_model: OptimizationModel,
    objectives_config: Dict[str, str],
    constraints_config: Dict[str, ConstraintSpec],
    plugin: DomainPlugin
) -> Optional[Tuple[Any, OptimizationProblem]]:
    """
    Render the solver selection and dynamic parameter configuration UI.

    This panel allows the user to:
        1. Select a solver from the registry.
        2. Adjust its hyperparameters using dynamically generated widgets.
        3. Run the optimisation (which returns a solver instance and problem).
        4. Save the generated Pareto front to the session state.

    Args:
        selected_model: The current OptimizationModel.
        objectives_config: Mapping of objective names to 'max' or 'min'.
        constraints_config: Mapping of attribute names to ConstraintSpec.
        plugin: The active domain plugin (for context, unused directly).

    Returns:
        A tuple (solver_instance, problem_instance) if the optimisation is
        triggered, otherwise None.
    """
    registered_solvers = get_registered_solvers()
    if not registered_solvers:
        st.error("No solvers found in the registry!")
        return None

    solver_names = list(registered_solvers.keys())
    selected_solver_name = st.selectbox(
        "Select Solver",
        options=solver_names,
        index=0,
        help="Choose the optimisation algorithm to run."
    )

    solver_cls = get_solver_class(selected_solver_name)
    param_kwargs = {}

    if solver_cls and hasattr(solver_cls, "get_parameter_schema"):
        schema = solver_cls.get_parameter_schema()
        if schema:
            st.markdown("##### Hyperparameters")
            for spec in schema:
                param_kwargs[spec.name] = render_parameter_widget(
                    spec=spec,
                    key=f"param_{selected_solver_name}_{spec.name}"
                )

    # --- Pareto name input (user can type anything) ---
    has_pareto = st.session_state.get("last_pareto_df") is not None
    pareto_name = ""
    if has_pareto:
        pareto_name = st.text_input("Pareto Name", value="", key="save_pareto_name")

    # --- Run button (primary action) ---
    run_clicked = st.button("🚀 Run Optimization", type="primary", use_container_width=True, key="btn_run_optimization")

    # --- Save button (secondary action, below run) ---
    save_clicked = False
    if has_pareto:
        save_clicked = st.button("💾 Save Pareto Front", use_container_width=True, type="secondary", key="btn_save_pareto")

    # --- Logic 1: Run optimisation ---
    if run_clicked:
        if not objectives_config:
            st.error("⚠️ Please configure at least one objective before running.")
            return None

        solver_instance = solver_cls(**param_kwargs)

        # Reset previous results
        st.session_state["last_problem"] = None
        st.session_state["last_pareto_front"] = None
        st.session_state["last_pareto_df"] = None
        st.session_state["last_solver"] = None
        st.session_state["last_execution_time"] = 0.0

        problem_instance = OptimizationProblem(
            model=selected_model,
            objectives=objectives_config,
            constraints=constraints_config,
        )

        st.session_state["last_solver_name"] = (
            solver_instance.get_name() if hasattr(solver_instance, "get_name")
            else solver_instance.__class__.__name__
        )

        return solver_instance, problem_instance

    # --- Logic 2: Save Pareto front ---
    if save_clicked and has_pareto:
        if not pareto_name.strip():
            pareto_name = f"Pareto Run #{len(st.session_state.get('saved_pareto_fronts', [])) + 1}"

        problem = st.session_state.get("last_problem")
        if problem is None:
            problem = OptimizationProblem(
                model=selected_model,
                objectives=objectives_config,
                constraints=constraints_config,
            )

        df_to_save = st.session_state["last_pareto_df"].copy()
        decision_cols = [col for col in df_to_save.columns if col.startswith("req_")]
        front = ParetoFront.from_dataframe(df_to_save, decision_cols=decision_cols)

        metadata = {
            "execution_time": st.session_state.get("last_execution_time", 0.0),
            "solver_name": st.session_state.get("last_solver_name", "N/A"),
            "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "decision_cols": decision_cols,
            "objective_cols": [col for col in df_to_save.columns
                               if col not in decision_cols
                               and col not in ["id", "selected_ids", "selected_ids_str", "is_feasible"]],
        }

        saved_set = problem.add_pareto_front(
            front=front,
            name=pareto_name,
            category="pareto_front",
            algorithm=st.session_state.get("last_solver_name", "N/A"),
            metadata=metadata,
        )

        if "saved_pareto_fronts" not in st.session_state:
            st.session_state["saved_pareto_fronts"] = []
        st.session_state["saved_pareto_fronts"].append({
            "id": saved_set.id,
            "name": saved_set.name,
            "df": df_to_save,
            "decision_cols": decision_cols,
            "objective_cols": metadata["objective_cols"],
            "execution_time": metadata["execution_time"],
            "solver_name": metadata["solver_name"],
            "created_at": metadata["created_at"],
            "objectives_config": objectives_config.copy(),
            "constraints_config": {
                attr: {
                    "operator": spec.operator.value if hasattr(spec, "operator") and hasattr(spec.operator, "value") else str(spec.operator),
                    "value": spec.value,
                    "attribute": getattr(spec, "attribute", attr),
                } if hasattr(spec, "operator") else spec
                for attr, spec in constraints_config.items()
            },
        })

        st.session_state["last_problem"] = problem
        st.success(f"Saved Pareto front '{pareto_name}' ({len(df_to_save)} solutions)!")
        st.rerun()

    return None