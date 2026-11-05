# Location: tests/test_evaluation_rules.py
"""
Tests to verify that the SolutionEvaluator respects the evaluation_rules
defined in the model's attribute definitions.

These tests cover sum, max, and mean evaluation rules, edge cases such as
empty selections and single-item selections, fallback behaviour, and
integration with the OptimizationProblem.
"""

import pytest
from src.domain.core.item import DecisionItem
from src.domain.core.model import OptimizationModel
from src.domain.core.definitions import AttributeDefinition
from src.solving.core.evaluator import SolutionEvaluator
from src.solving.core.problem import OptimizationProblem


@pytest.fixture
def model_with_evaluation_rules():
    """
    Create a model with three attributes using different evaluation rules:
        - 'effort': 'sum' (summation)
        - 'max_time': 'max' (maximum value)
        - 'avg_risk': 'mean' (average value)

    Returns:
        OptimizationModel instance with predefined items and attribute definitions.
    """
    # Define attribute definitions
    attr_defs = {
        "effort": AttributeDefinition(
            name="effort",
            type="scalar",
            evaluation_rule="sum",
            coupling_rule="sum"
        ),
        "max_time": AttributeDefinition(
            name="max_time",
            type="scalar",
            evaluation_rule="max",
            coupling_rule="max"
        ),
        "avg_risk": AttributeDefinition(
            name="avg_risk",
            type="scalar",
            evaluation_rule="mean",
            coupling_rule="mean"
        ),
    }

    # Create items with attribute values
    items = {
        "R1": DecisionItem(
            id="R1",
            description="Req 1",
            attributes={"effort": 5, "max_time": 10, "avg_risk": 3}
        ),
        "R2": DecisionItem(
            id="R2",
            description="Req 2",
            attributes={"effort": 8, "max_time": 7, "avg_risk": 5}
        ),
        "R3": DecisionItem(
            id="R3",
            description="Req 3",
            attributes={"effort": 3, "max_time": 15, "avg_risk": 2}
        ),
    }

    model = OptimizationModel(
        name="EvaluationRulesTest",
        items=items,
        attribute_definitions=attr_defs,
        relationships={}
    )

    return model


def test_evaluation_rule_sum(model_with_evaluation_rules):
    """
    Verify that the 'sum' evaluation rule works correctly.

    The 'effort' attribute is summed across selected items.
    Expected: 5 + 8 + 3 = 16.
    """
    evaluator = SolutionEvaluator(model_with_evaluation_rules)
    selected_ids = {"R1", "R2", "R3"}

    effort = evaluator.evaluate_attribute(selected_ids, "effort")
    assert effort == 16.0


def test_evaluation_rule_max(model_with_evaluation_rules):
    """
    Verify that the 'max' evaluation rule works correctly.

    The 'max_time' attribute should take the maximum value among selected items.
    Expected: max(10, 7, 15) = 15.
    """
    evaluator = SolutionEvaluator(model_with_evaluation_rules)
    selected_ids = {"R1", "R2", "R3"}

    max_time = evaluator.evaluate_attribute(selected_ids, "max_time")
    assert max_time == 15.0


def test_evaluation_rule_mean(model_with_evaluation_rules):
    """
    Verify that the 'mean' evaluation rule works correctly.

    The 'avg_risk' attribute should be the arithmetic mean of the selected items.
    Expected: (3 + 5 + 2) / 3 = 10 / 3 ≈ 3.333...
    """
    evaluator = SolutionEvaluator(model_with_evaluation_rules)
    selected_ids = {"R1", "R2", "R3"}

    avg_risk = evaluator.evaluate_attribute(selected_ids, "avg_risk")
    assert abs(avg_risk - 3.3333333333333335) < 1e-6


def test_evaluation_rule_with_empty_selection(model_with_evaluation_rules):
    """
    Verify that evaluating with an empty selection returns 0.0 for all attributes.
    """
    evaluator = SolutionEvaluator(model_with_evaluation_rules)
    selected_ids = set()

    effort = evaluator.evaluate_attribute(selected_ids, "effort")
    assert effort == 0.0


def test_evaluation_rule_with_single_item(model_with_evaluation_rules):
    """
    Verify that evaluation with a single item returns that item's attribute values
    directly (sum, max, and mean all reduce to the same value).
    """
    evaluator = SolutionEvaluator(model_with_evaluation_rules)
    selected_ids = {"R2"}

    # Effort: only R2 (8)
    effort = evaluator.evaluate_attribute(selected_ids, "effort")
    assert effort == 8.0

    # Max time: only R2 (7)
    max_time = evaluator.evaluate_attribute(selected_ids, "max_time")
    assert max_time == 7.0


def test_evaluation_rule_fallback_to_sum():
    """
    Verify that if no evaluation_rule is defined for an attribute, the default
    rule 'sum' is used.
    """
    # Model without attribute definitions
    items = {
        "R1": DecisionItem(id="R1", attributes={"cost": 10}),
        "R2": DecisionItem(id="R2", attributes={"cost": 20}),
    }
    model = OptimizationModel(name="FallbackTest", items=items)

    evaluator = SolutionEvaluator(model)
    selected_ids = {"R1", "R2"}

    # Should sum: 10 + 20 = 30
    cost = evaluator.evaluate_attribute(selected_ids, "cost")
    assert cost == 30.0


def test_evaluation_with_optimization_problem(model_with_evaluation_rules):
    """
    Integration test: verify that the OptimizationProblem correctly uses the
    evaluator to create a solution with the proper objective values.

    For selected items R1 and R3:
        - effort: 5 + 3 = 8
        - max_time: max(10, 15) = 15
        - avg_risk: (3 + 2) / 2 = 2.5
    """
    evaluator = SolutionEvaluator(model_with_evaluation_rules)
    selected_ids = {"R1", "R3"}

    # Create a simple problem
    problem = OptimizationProblem(
        model=model_with_evaluation_rules,
        objectives={"effort": "min", "max_time": "min", "avg_risk": "min"}
    )

    # Create the solution
    solution = evaluator.create_solution("sol_1", selected_ids, problem)

    # Verify objectives
    assert solution.objectives["effort"] == 8.0   # 5 + 3
    assert solution.objectives["max_time"] == 15.0  # max(10, 15) = 15
    assert abs(solution.objectives["avg_risk"] - 2.5) < 1e-6  # (3 + 2) / 2 = 2.5