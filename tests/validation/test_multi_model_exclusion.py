# tests/validation/test_multi_model_execution.py

import json

from src.application.use_cases.run_optimization import (
    run_optimization_models,
)

from src.domain.core.loader import GenericJSONLoader
from src.domain.core.pipeline import PreprocessingEngine
from src.domain.plugins.nrp import NRPPlugin

from src.solving.algorithms.brute_force import (
    BruteForceSolver,
)


def test_run_optimization_models_executes_all_branches():
    """
    Validate execution of all exclusion-generated subproblems.
    """

    case = {
        "name": "multi_model_execution",

        "attribute_definitions": {
            "value": {
                "type": "scalar",
                "evaluation_rule": "sum",
            },
            "cost": {
                "type": "scalar",
                "evaluation_rule": "sum",
            },
        },

        "requirements": [
            {
                "id": "r1",
                "attributes": {
                    "value": 10,
                    "cost": 5,
                },
            },
            {
                "id": "r2",
                "attributes": {
                    "value": 15,
                    "cost": 8,
                },
            },
            {
                "id": "r3",
                "attributes": {
                    "value": 12,
                    "cost": 4,
                },
            },
            {
                "id": "r4",
                "attributes": {
                    "value": 20,
                    "cost": 15,
                },
            },
        ],

        "relationships": {
            "precedences": [],
            "couplings": [],
            "exclusions": [
                {"req1": "r1", "req2": "r2"},
                {"req1": "r3", "req2": "r4"},
            ],
            "value_dependencies": [],
        },
    }

    base_model = GenericJSONLoader.load(
        json.dumps(case),
        NRPPlugin,
    )

    processed_models = PreprocessingEngine.process(
        initial_model=base_model,
        pipeline_options={},
    )

    # 2 exclusions => 4 branches
    assert len(processed_models) == 4

    def solver_factory():
        return BruteForceSolver(
            max_evaluations=10000
        )

    result = run_optimization_models(
        models=processed_models,
        objectives={
            "value": "max",
            "cost": "min",
        },
        constraints={},
        solver_factory=solver_factory,
    )

    assert len(result.partial_results) == 4

    assert len(
        result.pareto_front.solutions
    ) > 0

    assert not result.pareto_df.empty

    assert result.execution_time >= 0.0

    assert all(
        len(
            partial_result.pareto_front.solutions
        ) > 0
        for partial_result
        in result.partial_results
    )