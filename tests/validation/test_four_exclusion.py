# Location: tests/validation/test_four_exclusions.py

"""
Bounded validation checks for four independent exclusions.

The validation case contains four independent exclusion pairs:

    r1 XOR r2
    r3 XOR r4
    r5 XOR r6
    r7 XOR r8

Each exclusion introduces two alternative branches. Therefore, the
complete preprocessing pipeline must generate:

    2^4 = 16 subproblems

Every resulting subproblem must retain exactly one requirement from
each exclusion pair. The tests also verify that all sixteen subproblems
can be solved by the headless multi-model execution use case.

The operational UI limit of three exclusions is tested separately.
This module validates the semantic capability of the preprocessing and
application layers, not whether the UI exposes the "All" option.
"""

import itertools
import json
from pathlib import Path

from src.application.use_cases.run_optimization import (
    run_optimization_models,
)
from src.domain.core.loader import GenericJSONLoader
from src.domain.core.pipeline import PreprocessingEngine
from src.domain.plugins.nrp import NRPPlugin
from src.solving.algorithms.brute_force import (
    BruteForceSolver,
)


# Change this path only if the validation cases are stored elsewhere.
FOUR_EXCLUSIONS_CASE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "instances"
    / "validation"
    / "four_exclusions.json"
)


EXCLUSION_PAIRS = (
    ("r1", "r2"),
    ("r3", "r4"),
    ("r5", "r6"),
    ("r7", "r8"),
)


def _load_case() -> dict:
    """
    Load the controlled four-exclusion validation case.
    """

    assert FOUR_EXCLUSIONS_CASE_PATH.exists(), (
        "The validation case was not found at "
        f"{FOUR_EXCLUSIONS_CASE_PATH}."
    )

    return json.loads(
        FOUR_EXCLUSIONS_CASE_PATH.read_text(
            encoding="utf-8"
        )
    )


def _load_and_preprocess():
    """
    Load the JSON case and execute the complete preprocessing pipeline.
    """

    case = _load_case()

    base_model = GenericJSONLoader.load(
        json.dumps(case),
        NRPPlugin,
    )

    processed_models = PreprocessingEngine.process(
        initial_model=base_model,
        pipeline_options={},
    )

    return case, base_model, processed_models


def _branch_signature(model) -> tuple[str, ...]:
    """
    Return the single surviving requirement from each exclusion pair.

    The signature provides a stable representation of one branch.
    """

    surviving_ids = set(model.items)

    signature = []

    for first_id, second_id in EXCLUSION_PAIRS:
        surviving_pair = (
            surviving_ids
            & {first_id, second_id}
        )

        assert len(surviving_pair) == 1, (
            "Each subproblem must retain exactly one requirement "
            f"from exclusion pair ({first_id}, {second_id}), but "
            f"the surviving set was {sorted(surviving_pair)}."
        )

        signature.append(
            next(iter(surviving_pair))
        )

    return tuple(signature)


def test_four_exclusions_json_has_expected_structure():
    """
    The JSON case must contain four independent and valid exclusions.
    """

    case = _load_case()

    requirements = {
        requirement["id"]
        for requirement in case["requirements"]
    }

    exclusions = case[
        "relationships"
    ]["exclusions"]

    assert len(requirements) == 8
    assert len(exclusions) == 4

    parsed_pairs = {
        (
            exclusion["req1"],
            exclusion["req2"],
        )
        for exclusion in exclusions
    }

    assert parsed_pairs == set(
        EXCLUSION_PAIRS
    )

    for first_id, second_id in parsed_pairs:
        assert first_id in requirements
        assert second_id in requirements
        assert first_id != second_id


def test_four_independent_exclusions_generate_sixteen_branches():
    """
    Four independent exclusions must generate exactly sixteen branches.
    """

    _case, _base_model, processed_models = (
        _load_and_preprocess()
    )

    assert len(processed_models) == 16


def test_sixteen_branches_cover_all_exclusion_combinations():
    """
    The generated branches must correspond exactly to the Cartesian
    product of the four independent exclusion choices.
    """

    _case, _base_model, processed_models = (
        _load_and_preprocess()
    )

    actual_signatures = {
        _branch_signature(model)
        for model in processed_models
    }

    expected_signatures = set(
        itertools.product(
            *EXCLUSION_PAIRS
        )
    )

    assert len(actual_signatures) == 16
    assert actual_signatures == expected_signatures


def test_every_branch_contains_four_requirements_and_no_exclusions():
    """
    Every branch must retain exactly one requirement from each pair.

    Since the exclusion choices have already been resolved, processed
    models must not retain unresolved exclusion relationships.
    """

    _case, _base_model, processed_models = (
        _load_and_preprocess()
    )

    for model in processed_models:
        surviving_ids = set(model.items)

        assert len(surviving_ids) == 4

        assert model.relationships.get(
            "exclusions",
            [],
        ) == []

        for first_id, second_id in EXCLUSION_PAIRS:
            assert (
                len(
                    surviving_ids
                    & {first_id, second_id}
                )
                == 1
            )


def test_all_sixteen_subproblems_can_be_solved_headlessly():
    """
    The headless multi-model use case must solve all sixteen branches.

    This test deliberately bypasses the UI limit of three exclusions.
    The limit controls whether the UI exposes automatic "All" execution;
    it does not limit the semantic capabilities of the preprocessing or
    application layers.
    """

    _case, _base_model, processed_models = (
        _load_and_preprocess()
    )

    solver_instances_created = 0

    def solver_factory():
        nonlocal solver_instances_created

        solver_instances_created += 1

        return BruteForceSolver(
            max_evaluations=1000
        )

    result = run_optimization_models(
        models=processed_models,
        objectives={
            "value": "max",
        },
        constraints={},
        solver_factory=solver_factory,
    )

    assert solver_instances_created == 16
    assert len(result.partial_results) == 16

    assert all(
        len(
            partial_result
            .pareto_front
            .solutions
        )
        > 0
        for partial_result
        in result.partial_results
    )

    assert len(
        result.pareto_front.solutions
    ) > 0

    assert not result.pareto_df.empty
    assert result.execution_time >= 0.0

    assert (
        result.pareto_front.objective_directions
        == {
            "value": "max",
        }
    )


def test_global_front_selects_best_exclusion_combination():
    """
    With positive additive values and a single maximisation objective,
    the global optimum must select the highest-value requirement from
    every exclusion pair.

    Requirement values in four_exclusion.json are:

        r1=1, r2=2
        r3=3, r4=4
        r5=5, r6=6
        r7=7, r8=8

    Therefore, the unique global optimum is:

        {r2, r4, r6, r8}

    with total value:

        2 + 4 + 6 + 8 = 20
    """

    _case, _base_model, processed_models = (
        _load_and_preprocess()
    )

    def solver_factory():
        return BruteForceSolver(
            max_evaluations=1000
        )

    result = run_optimization_models(
        models=processed_models,
        objectives={
            "value": "max",
        },
        constraints={},
        solver_factory=solver_factory,
    )

    global_solutions = (
        result.pareto_front.solutions
    )

    assert len(global_solutions) == 1

    global_solution = global_solutions[0]

    assert set(
        global_solution.selected_ids
    ) == {
        "r2",
        "r4",
        "r6",
        "r8",
    }

    assert float(
        global_solution.objectives["value"]
    ) == 20.0

    details = dict(
        getattr(
            global_solution,
            "evaluation_details",
            {},
        )
        or {}
    )

    assert "subproblem_branch" in details
    assert "subproblem_index" in details
    assert "original_solution_id" in details