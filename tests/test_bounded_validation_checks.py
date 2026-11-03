"""
Bounded software-validation checks.

These tests exercise small interaction-bearing formulations and compare
their behaviour before and after preprocessing.

The goal is not algorithm benchmarking, but validation of the framework's
transformation and evaluation mechanisms on controlled instances.
"""

import json

from src.domain.core.loader import GenericJSONLoader
from src.domain.core.pipeline import PreprocessingEngine
from src.domain.plugins.nrp import NRPPlugin
from src.solving.core.evaluator import SolutionEvaluator


def _evaluate_before_after(case, selected_ids, attribute):
    """
    Evaluate the same release before and after preprocessing.
    """

    base_model = GenericJSONLoader.load(
        json.dumps(case),
        NRPPlugin,
    )

    before = SolutionEvaluator(
        base_model
    ).evaluate_attribute(
        set(selected_ids),
        attribute,
    )

    processed_models = PreprocessingEngine.process(
        initial_model=base_model,
        pipeline_options={},
    )

    assert len(processed_models) == 1

    processed_model = processed_models[0]

    after = SolutionEvaluator(
        processed_model
    ).evaluate_attribute(
        set(processed_model.items.keys()),
        attribute,
    )

    return before, after


def test_coupling_multiplier_subset():
    """
    Reviewer-2 coupling edge case.

    (10+20)*2 +100 = 160
    """

    case = {
        "name": "coupling_multiplier_subset",
        "attribute_definitions": {
            "value": {
                "type": "scalar",
                "coupling_rule": "sum",
                "evaluation_rule": "sum",
            }
        },
        "requirements": [
            {"id": "r1", "attributes": {"value": 10}},
            {"id": "r2", "attributes": {"value": 20}},
            {"id": "r3", "attributes": {"value": 100}},
        ],
        "relationships": {
            "precedences": [],
            "couplings": [
                {"req1": "r1", "req2": "r2"},
                {"req1": "r2", "req2": "r3"},
            ],
            "exclusions": [],
            "value_dependencies": [
                {
                    "attribute": "value",
                    "reqs": ["r1", "r2"],
                    "effect": "multiplier",
                    "factor": 2.0,
                }
            ],
        },
    }

    before, after = _evaluate_before_after(
        case,
        ["r1", "r2", "r3"],
        "value",
    )

    assert before == 160.0
    assert after == 160.0
    assert before == after


def test_coupling_delta_subset():
    """
    Delta absorbed through coupling.

    10 + 20 + 100 + 5 = 135
    """

    case = {
        "name": "coupling_delta_subset",
        "attribute_definitions": {
            "value": {
                "type": "scalar",
                "coupling_rule": "sum",
                "evaluation_rule": "sum",
            }
        },
        "requirements": [
            {"id": "r1", "attributes": {"value": 10}},
            {"id": "r2", "attributes": {"value": 20}},
            {"id": "r3", "attributes": {"value": 100}},
        ],
        "relationships": {
            "precedences": [],
            "couplings": [
                {"req1": "r1", "req2": "r2"},
                {"req1": "r2", "req2": "r3"},
            ],
            "exclusions": [],
            "value_dependencies": [
                {
                    "attribute": "value",
                    "reqs": ["r1", "r2"],
                    "effect": "delta",
                    "factor": 5.0,
                }
            ],
        },
    }

    before, after = _evaluate_before_after(
        case,
        ["r1", "r2", "r3"],
        "value",
    )

    assert before == 135.0
    assert after == 135.0
    assert before == after


def test_multiplier_reduction():
    """
    Reduction multiplier.

    (10+20)*0.5 +100 = 115
    """

    case = {
        "name": "multiplier_reduction",
        "attribute_definitions": {
            "value": {
                "type": "scalar",
                "coupling_rule": "sum",
                "evaluation_rule": "sum",
            }
        },
        "requirements": [
            {"id": "r1", "attributes": {"value": 10}},
            {"id": "r2", "attributes": {"value": 20}},
            {"id": "r3", "attributes": {"value": 100}},
        ],
        "relationships": {
            "precedences": [],
            "couplings": [
                {"req1": "r1", "req2": "r2"},
                {"req1": "r2", "req2": "r3"},
            ],
            "exclusions": [],
            "value_dependencies": [
                {
                    "attribute": "value",
                    "reqs": ["r1", "r2"],
                    "effect": "multiplier",
                    "factor": 0.5,
                }
            ],
        },
    }

    before, after = _evaluate_before_after(
        case,
        ["r1", "r2", "r3"],
        "value",
    )

    assert before == 115.0
    assert after == 115.0
    assert before == after


def test_implication_preserved():
    """
    Small implication-bearing formulation.
    """

    case = {
        "name": "implication_case",
        "attribute_definitions": {
            "value": {
                "type": "scalar",
                "evaluation_rule": "sum"
            }
        },
        "requirements": [
            {"id": "r1", "attributes": {"value": 10}},
            {"id": "r2", "attributes": {"value": 20}},
        ],
        "relationships": {
            "precedences": [
                {"req1": "r1", "req2": "r2"}
            ],
            "couplings": [],
            "exclusions": [],
            "value_dependencies": []
        }
    }

    base_model = GenericJSONLoader.load(
        json.dumps(case),
        NRPPlugin,
    )

    processed_models = PreprocessingEngine.process(
        initial_model=base_model,
        pipeline_options={},
    )

    processed = processed_models[0]

    assert len(
        processed.relationships["precedences"]
    ) == 1