"""
Validation of exclusion-based execution limits.
"""

from src.application.optimization_limits import (
    can_optimise_all_exclusion_subproblems,
)


def test_all_execution_accepts_up_to_three_exclusions():
    """
    Automatic execution of all exclusion-generated
    subproblems is allowed up to the configured limit.
    """

    assert can_optimise_all_exclusion_subproblems(0)
    assert can_optimise_all_exclusion_subproblems(1)
    assert can_optimise_all_exclusion_subproblems(2)
    assert can_optimise_all_exclusion_subproblems(3)


def test_all_execution_rejects_more_than_three_exclusions():
    """
    Automatic execution of all exclusion-generated
    subproblems is disabled above the configured limit.
    """

    assert not can_optimise_all_exclusion_subproblems(4)
    assert not can_optimise_all_exclusion_subproblems(5)