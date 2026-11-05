# tests/test_solvers.py
"""
Unit tests for the optimisation solvers.

These tests verify that each solver (Greedy, Random Search, Brute Force,
NSGA‑II, MILP, Simulated Annealing) can generate a non‑empty Pareto front
of feasible solutions for a small test problem. They also check that each
solver provides a valid parameter schema.
"""

import pytest
from src.domain.core.model import OptimizationModel
from src.domain.core.item import DecisionItem
from src.domain.core.definitions import AttributeDefinition
from src.solving.core.problem import OptimizationProblem, ConstraintSpec, ThresholdOperator
from src.solving.algorithms.greedy import GreedySolver
from src.solving.algorithms.random import RandomSearchSolver
from src.solving.algorithms.brute_force import BruteForceSolver
from src.solving.algorithms.nsga2 import NSGA2Solver
from src.solving.algorithms.milp import MILPSolver
from src.solving.algorithms.simulated_annealing import SimulatedAnnealingSolver

 

@pytest.fixture
def small_model():
    """Create a small model with 4 items and two objectives (cost, value)."""
    items = {
        "A": DecisionItem("A", "Item A", {"cost": 10.0, "value": 50.0}),
        "B": DecisionItem("B", "Item B", {"cost": 20.0, "value": 80.0}),
        "C": DecisionItem("C", "Item C", {"cost": 15.0, "value": 40.0}),
        "D": DecisionItem("D", "Item D", {"cost": 30.0, "value": 100.0}),
    }
    attr_defs = {
        "cost": AttributeDefinition(name="cost", type="scalar", evaluation_rule="sum"),
        "value": AttributeDefinition(name="value", type="scalar", evaluation_rule="sum"),
    }
    return OptimizationModel(
        name="TestModel",
        items=items,
        attribute_definitions=attr_defs,
        relationships={"precedences": [], "couplings": [], "exclusions": []}
    )


@pytest.fixture
def small_problem(small_model):
    """Create an optimisation problem with the small model."""
    return OptimizationProblem(
        model=small_model,
        objectives={"value": "max", "cost": "min"},
        constraints={"cost": ConstraintSpec(operator=ThresholdOperator.LESS_EQUAL, value=50.0)}
    )


def test_greedy_solver(small_problem):
    """Test that the Greedy solver produces a non‑empty feasible front."""
    solver = GreedySolver(weight_samples=5)
    front = solver.solve(small_problem)
    assert len(front) > 0
    # Verify that all solutions are feasible
    for sol in front.solutions:
        assert sol.is_feasible is True


def test_random_search_solver(small_problem):
    """Test that the Random Search solver produces a non‑empty feasible front."""
    solver = RandomSearchSolver(num_samples=50)
    front = solver.solve(small_problem)
    assert len(front) > 0
    for sol in front.solutions:
        assert sol.is_feasible is True


def test_brute_force_solver(small_problem):
    """Test that the Brute Force solver produces a non‑empty feasible front."""
    solver = BruteForceSolver(max_evaluations=1000)
    front = solver.solve(small_problem)
    assert len(front) > 0
    for sol in front.solutions:
        assert sol.is_feasible is True


def test_nsga2_solver(small_problem):
    """Test that the NSGA‑II solver produces a non‑empty feasible front."""
    solver = NSGA2Solver(pop_size=20, generations=10)
    front = solver.solve(small_problem)
    assert len(front) > 0
    for sol in front.solutions:
        assert sol.is_feasible is True


def test_milp_solver(small_problem):
    """Test that the MILP solver produces a non‑empty feasible front."""
    solver = MILPSolver(weight_samples=3, time_limit_per_sweep=1.0)
    front = solver.solve(small_problem)
    # MILP may not find solutions if the model is too restrictive, but here it should.
    assert len(front) > 0
    for sol in front.solutions:
        assert sol.is_feasible is True


def test_simulated_annealing_solver(small_problem):
    """Test that the Simulated Annealing solver produces a non‑empty feasible front."""
    solver = SimulatedAnnealingSolver(iterations=200, weight_samples=3)
    front = solver.solve(small_problem)
    assert len(front) > 0
    for sol in front.solutions:
        assert sol.is_feasible is True


def test_solver_parameter_schema():
    """
    Verify that each solver provides a valid parameter schema.

    The schema should be a non‑empty list of ParameterSpec objects.
    """
    for solver_cls in [GreedySolver, RandomSearchSolver, BruteForceSolver, NSGA2Solver, MILPSolver, SimulatedAnnealingSolver]:
        schema = solver_cls.get_parameter_schema()
        assert isinstance(schema, list)
        # Each solver must have at least one parameter
        assert len(schema) > 0



@pytest.fixture
def precedence_model():
    """
    A requires B (A -> B): selecting A without B must be infeasible.
    B has a much better value/cost ratio than A, so an unconstrained
    maximiser would pick B alone if the direction were wrong.
    """
    items = {
        "A": DecisionItem("A", "Item A", {"cost": 5.0, "value": 5.0}),
        "B": DecisionItem("B", "Item B", {"cost": 5.0, "value": 5.0}),
    }
    attr_defs = {
        "cost": AttributeDefinition(name="cost", type="scalar", evaluation_rule="sum"),
        "value": AttributeDefinition(name="value", type="scalar", evaluation_rule="sum"),
    }
    return OptimizationModel(
        name="PrecedenceModel",
        items=items,
        attribute_definitions=attr_defs,
        relationships={
            "precedences": [{"from": "A", "to": "B"}],  # selecting A requires B
            "couplings": [],
            "exclusions": [],
        },
    )


def test_milp_respects_implication_direction(precedence_model):
    """
    Regression test for the implication constraint sign in MILPSolver.
    With A -> B, every feasible/returned release that contains A must
    also contain B. A release with A but without B must never appear in
    the returned Pareto front (it would if the constraint were built
    backwards as x_B <= x_A instead of x_A <= x_B).
    """
    problem = OptimizationProblem(
        model=precedence_model,
        objectives={"value": "max", "cost": "min"},
        constraints={"cost": ConstraintSpec(operator=ThresholdOperator.LESS_EQUAL, value=100.0)},
    )
    solver = MILPSolver(weight_samples=5, time_limit_per_sweep=1.0)
    front = solver.solve(problem)

    for sol in front.solutions:
        selected = set(sol.selected_ids)
        if "A" in selected:
            assert "B" in selected, (
                "MILP returned a release selecting A without B, violating "
                "the declared implication A -> B (selecting A requires B)."
            )