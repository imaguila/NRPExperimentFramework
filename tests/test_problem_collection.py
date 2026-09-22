# Location: tests/test_problem_collection.py
"""
Tests for the OptimizationProblem and related collection functionality.

These tests verify that Pareto fronts are correctly added to the problem
and synchronised with the model, that solving and storing works,
that problem metadata is preserved, and that round‑trip serialisation
and objective directions are handled correctly.
"""

import pandas as pd
import pytest
import streamlit as st

from src.domain.core.model import OptimizationModel
from src.domain.core.paretofront import ParetoFront
from src.domain.core.solution import Solution
from src.domain.core.item import DecisionItem
from src.interface.panels.comparison_panel import render_comparison_panel
from src.solving.core.problem import OptimizationProblem, ConstraintSpec, ThresholdOperator
from src.solving.algorithms.greedy import GreedySolver


@pytest.fixture
def setup_problem():
    """Create a simple model and optimisation problem for testing."""
    items = {
        "1": DecisionItem(id="1", attributes={"satisfaction": 10, "effort": 2}),
        "2": DecisionItem(id="2", attributes={"satisfaction": 20, "effort": 5}),
    }
    model = OptimizationModel(name="Sync Model", items=items)
    problem = OptimizationProblem(
        model=model,
        objectives={"satisfaction": "max", "effort": "min"}
    )
    return model, problem


def test_problem_add_pareto_front_syncs_with_model(setup_problem):
    """
    Test that adding a Pareto front to the problem also synchronises it
    with the underlying model's solutions collection.
    """
    model, problem = setup_problem
    sol = Solution(id="s1", selected_ids={"1"}, objectives={"satisfaction": 10, "effort": 2})
    front = ParetoFront([sol])

    # Store in the problem
    s_set = problem.add_pareto_front(front, name="Problem Run", algorithm="TestAlg")

    # Verify it exists in both the problem and the model
    assert len(problem.solutions_collection) == 1
    assert len(model.solutions_collection) == 1
    assert model.solutions_collection[s_set.id] == s_set


def test_problem_solve_and_store(setup_problem):
    """
    Test that solving a problem with a solver and storing the result
    works correctly and that the algorithm name is recorded.
    """
    model, problem = setup_problem
    solver = GreedySolver()

    front, s_set = problem.solve_and_store(solver, name="Greedy Run")

    assert isinstance(front, ParetoFront)
    assert len(problem.solutions_collection) == 1
    assert len(model.solutions_collection) == 1
    assert problem.solutions_collection[s_set.id] == s_set
    assert s_set.algorithm == "GreedySolver"


def test_problem_pareto_metadata_contains_optimization_config(setup_problem):
    """
    Test that the metadata stored with a Pareto front includes the full
    problem configuration (objectives, constraints, etc.).
    """
    model, problem = setup_problem
    problem.constraints = {
        "effort": ConstraintSpec(operator=ThresholdOperator.LESS_EQUAL, value=5, attribute="effort")
    }
    problem.primary_constraint_attr = "effort"

    sol = Solution(id="s1", selected_ids={"1"}, objectives={"satisfaction": 10, "effort": 2})
    front = ParetoFront([sol])

    s_set = problem.add_pareto_front(front, name="Configured Run", algorithm="TestAlg")

    assert s_set.metadata["problem"]
    assert s_set.metadata["problem"]["model_name"] == "Sync Model"
    assert s_set.metadata["problem"]["objectives"] == {"satisfaction": "max", "effort": "min"}
    assert s_set.metadata["problem"]["constraints"]["effort"]["value"] == 5
    assert s_set.metadata["problem"]["constraints"]["effort"]["operator"] == "<="


def test_render_comparison_panel_uses_problem_objective_directions(monkeypatch):
    """
    Test that the comparison panel correctly resolves objective directions
    from the problem instance.
    """
    df = pd.DataFrame([
        {"id": "a", "cost": 10.0, "value": 9.0, "req_1": 1, "req_2": 0},
        {"id": "b", "cost": 8.0, "value": 7.0, "req_1": 0, "req_2": 1},
    ])
    problem = OptimizationProblem(
        model=OptimizationModel(name="Compare Model", items={}),
        objectives={"cost": "min", "value": "max"},
    )
    st.session_state["saved_pareto_fronts"] = [{
        "name": "Front A",
        "df": df,
        "decision_cols": ["req_1", "req_2"],
        "objective_cols": ["cost", "value"],
        "objectives_config": {"cost": "min", "value": "max"},
    }]
    observed = {}

    monkeypatch.setattr(st, "multiselect", lambda *args, **kwargs: ["Front A"])
    monkeypatch.setattr(st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "download_button", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "plotly_chart", lambda *args, **kwargs: None)

    original_compute = ParetoFront.compute_metrics
    def fake_compute_metrics(self, *args, **kwargs):
        observed["directions"] = kwargs.get("directions")
        return {"Front Name": "Front A", "Solutions": 2, "Hypervolume": 1.0, "Spacing": 0.0, "Spread": 0.0}
    monkeypatch.setattr(ParetoFront, "compute_metrics", fake_compute_metrics)

    current_front = ParetoFront.from_dataframe(df=df, decision_cols=["req_1", "req_2"], objective_cols=["cost", "value"])
    render_comparison_panel(current_front=current_front, problem=problem)

    assert observed["directions"] == {"cost": "min", "value": "max"}
    monkeypatch.setattr(ParetoFront, "compute_metrics", original_compute)


def test_pareto_front_roundtrip_keeps_objective_directions():
    """
    Test that objective directions are preserved when a Pareto front is
    serialised to a dictionary and then reconstructed.
    """
    front = ParetoFront([
        Solution(id="a", selected_ids={"1"}, objectives={"cost": 10.0, "value": 9.0}, is_feasible=True),
        Solution(id="b", selected_ids={"2"}, objectives={"cost": 8.0, "value": 11.0}, is_feasible=True),
    ])
    front.set_objective_directions({"cost": "min", "value": "max"})

    payload = front.to_dict()
    recovered = ParetoFront.from_dict(payload)

    assert recovered.objective_directions == {"cost": "min", "value": "max"}


def test_nrp_synthetic_instance_export_roundtrip_keeps_structure():
    """
    Test that a synthetic NRP instance can be exported to a dictionary
    and the resulting structure has the expected items and attributes.
    """
    from src.domain.plugins.nrp import NRPPlugin
    model = NRPPlugin.generate_synthetic_instance(
        num_items=3,
        attribute_configs={
            "effort": {"distribution": "uniform", "min": 1, "max": 5, "is_integer": True},
            "value": {"distribution": "uniform", "min": 10, "max": 20, "is_integer": True},
        },
        value_dependencies=[],
        enable_precedences=False,
    )

    payload = NRPPlugin.export_to_dict(model)
    assert payload["name"].startswith("Synthetic_NRP_")
    assert len(payload["requirements"]) == 3
    assert "effort" in payload["requirements"][0]["attributes"]
    assert "value" in payload["requirements"][0]["attributes"]
    assert payload["relationships"]["value_dependencies"] == []


def test_pareto_front_from_dataframe_keeps_numeric_attributes_and_objective_directions():
    """
    Test that creating a ParetoFront from a DataFrame preserves numeric
    attributes and objective directions.
    """
    df = pd.DataFrame([
        {"id": "a", "req_1": 1, "req_2": 0, "cost": 10.0, "value": 9.0, "color": 5.0},
        {"id": "b", "req_1": 0, "req_2": 1, "cost": 8.0, "value": 11.0, "color": 3.0},
    ])

    front = ParetoFront.from_dataframe(
        df=df,
        decision_cols=["req_1", "req_2"],
        objective_cols=["cost", "value"],
        objective_directions={"cost": "min", "value": "max"},
    )

    assert front.objective_directions == {"cost": "min", "value": "max"}
    assert front.solutions[0].attributes["color"] == 5.0
    assert front.solutions[1].attributes["color"] == 3.0