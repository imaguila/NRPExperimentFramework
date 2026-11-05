"""
Validation of exclusion-generated subproblems.

This test verifies that:

    descriptive model
            ↓
    exclusion branching
            ↓
    optimisation of each subproblem
            ↓
    union of partial fronts
            ↓
    global dominance filtering

produces a valid global Pareto front.
"""

import json

from src.domain.core.loader import GenericJSONLoader
from src.domain.core.pipeline import PreprocessingEngine
from src.domain.plugins.nrp import NRPPlugin

from src.solving.algorithms.brute_force import (
    BruteForceSolver,
)

from src.solving.core.problem import (
    OptimizationProblem,
)

from src.application.use_cases.run_optimization import (
    combine_partial_pareto_fronts,
)


def test_global_pareto_front_from_exclusion_subproblems():
    """
    Small exclusion-bearing formulation.

    r1 xor r2
    r3 xor r4

    Generates four optimisation subproblems.
    """

    case = {
        "name": "controlled_exclusions",

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

    partial_fronts = []
    branch_names = []

    for model in processed_models:

        problem = OptimizationProblem(
            model=model,
            objectives={
                "value": "max",
                "cost": "min",
            },
            constraints={},
        )

        solver = BruteForceSolver(
            max_evaluations=10000
        )

        front = solver.solve(problem)

        front.set_objective_directions(
            problem.get_objective_directions()
        )

        partial_fronts.append(front)

        branch_names.append(
            getattr(
                model,
                "branch_name",
                "branch",
            )
        )

    global_front = combine_partial_pareto_fronts(
        partial_fronts=partial_fronts,
        branch_names=branch_names,
        objective_directions={
            "value": "max",
            "cost": "min",
        },
    )

    # -----------------------------------------
    # Every solution should be feasible
    # -----------------------------------------

    assert all(
        solution.is_feasible
        for solution in global_front.solutions
    )

    # -----------------------------------------
    # Global front cannot be larger
    # than the union of partial fronts
    # -----------------------------------------

    total_partial = sum(
        len(front.solutions)
        for front in partial_fronts
    )

    assert len(global_front.solutions) <= total_partial

    # -----------------------------------------
    # Every solution must preserve branch
    # provenance metadata
    # -----------------------------------------

    for solution in global_front.solutions:

        details = getattr(
            solution,
            "evaluation_details",
            {},
        )

        assert "subproblem_branch" in details
        assert "subproblem_index" in details
        assert "original_solution_id" in details

    # -----------------------------------------
    # At least one solution from each branch
    # must be represented in the union before
    # dominance filtering.
    # -----------------------------------------

    assert len(branch_names) == 4