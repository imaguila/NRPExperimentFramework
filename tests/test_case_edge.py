"""
Regression tests for preprocessing semantic preservation.

These tests verify that coupling resolution and value-dependency
processing preserve the original semantics of the formulation.

The tests focus on interactions defined over requirements that
become part of the same coupling component after preprocessing.

The objective values computed on the original descriptive model
must remain equivalent after preprocessing.
"""

import json

from src.domain.core.loader import GenericJSONLoader
from src.domain.core.pipeline import PreprocessingEngine
from src.domain.plugins.nrp import NRPPlugin
from src.solving.core.evaluator import SolutionEvaluator


def _evaluate_before_after(case, selected_original):
    """
    Evaluate the same release before and after preprocessing.
    """

    base_model = GenericJSONLoader.load(
        json.dumps(case),
        NRPPlugin,
    )

    evaluator_before = SolutionEvaluator(base_model)

    value_before = evaluator_before.evaluate_attribute(
        set(selected_original),
        "value",
    )

    processed_models = PreprocessingEngine.process(
        initial_model=base_model,
        pipeline_options={},
    )

    assert len(processed_models) == 1

    processed_model = processed_models[0]

    evaluator_after = SolutionEvaluator(
        processed_model
    )

    unified_ids = set(processed_model.items.keys())

    value_after = evaluator_after.evaluate_attribute(
        unified_ids,
        "value",
    )

    return value_before, value_after


def test_multiplier_subset_inside_larger_coupling_component():
    """
    Reviewer-2 edge case.

    r1 = 10
    r2 = 20
    r3 = 100

    coupling:
        r1-r2
        r2-r3

    multiplier:
        (r1,r2) * 2

    expected:
        (10+20)*2 + 100 = 160
    """

    case = {
        "name": "subset_multiplier",
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
    )

    assert before == 160.0
    assert after == 160.0
    assert before == after


def test_delta_subset_inside_larger_coupling_component():
    """
    Same structure as the reviewer case,
    but using a delta interaction.
    """

    case = {
        "name": "subset_delta",
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
    )

    assert before == 135.0
    assert after == 135.0
    assert before == after


def test_multiplier_reduction_inside_larger_component():
    """
    Reduction multiplier (<1).

    Expected:
        (10+20)*0.5 + 100 = 115
    """

    case = {
        "name": "subset_reduction",
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
    )

    assert before == 115.0
    assert after == 115.0
    assert before == after