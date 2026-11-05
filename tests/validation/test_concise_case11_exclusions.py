"""
Validation of exclusion branching using the Concise_Case_11
instance distributed with the replication package.

This test verifies that:

    r7 XOR r9

produces exactly two optimisation subproblems and that the
resulting variants preserve the expected requirements after
preprocessing.
"""

from pathlib import Path

from src.domain.core.loader import GenericJSONLoader
from src.domain.core.pipeline import PreprocessingEngine
from src.domain.plugins.nrp import NRPPlugin


CASE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "instances"
    / "paper"
    / "Concise_Case_11.json"
)


def test_concise_case11_generates_two_exclusion_branches():
    """
    Concise_Case_11 contains:

        r7 XOR r9

    therefore two branches must be generated.
    """

    raw_json = CASE_PATH.read_text(
        encoding="utf-8"
    )

    base_model = GenericJSONLoader.load(
        raw_json,
        NRPPlugin,
    )

    processed_models = PreprocessingEngine.process(
        initial_model=base_model,
        pipeline_options={},
    )

    assert len(processed_models) == 2

    branch_names = {
        getattr(
            model,
            "branch_name",
            "",
        )
        for model in processed_models
    }

    assert "Excl(r7)" in branch_names
    assert "Excl(r9)" in branch_names


def test_concise_case11_excl_r7_branch():
    """
    Branch Excl(r7).

    r7 must disappear.

    Since r7 is removed by exclusion,
    r9 remains available.
    """

    raw_json = CASE_PATH.read_text(
        encoding="utf-8"
    )

    base_model = GenericJSONLoader.load(
        raw_json,
        NRPPlugin,
    )

    processed_models = PreprocessingEngine.process(
        initial_model=base_model,
        pipeline_options={},
    )

    branch = next(
        model
        for model in processed_models
        if getattr(
            model,
            "branch_name",
            "",
        )
        == "Excl(r7)"
    )

    items = set(branch.items.keys())

    assert "r7" not in items

    # Alternative branch survives
    assert "r9" in items

    # Dependency chain remains available
    assert "r10" in items
    assert "r12" in items


def test_concise_case11_excl_r9_branch():
    """
    Branch Excl(r9).

    r9 disappears because of the exclusion.

    Under the current framework semantics:

        r10 requires r9
        r12 requires r10

    therefore removing r9 triggers the
    transitive dependency cascade.
    """

    raw_json = CASE_PATH.read_text(
        encoding="utf-8"
    )

    base_model = GenericJSONLoader.load(
        raw_json,
        NRPPlugin,
    )

    processed_models = PreprocessingEngine.process(
        initial_model=base_model,
        pipeline_options={},
    )

    branch = next(
        model
        for model in processed_models
        if getattr(
            model,
            "branch_name",
            "",
        )
        == "Excl(r9)"
    )

    items = set(branch.items.keys())

    # Surviving alternative
    assert "r7" in items

    # Removed by exclusion
    assert "r9" not in items

    # Cascading removals
    assert "r10" not in items
    assert "r12" not in items