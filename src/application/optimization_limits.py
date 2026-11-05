MAX_EXCLUSIONS_FOR_ALL_SUBPROBLEMS = 3
ALL_SUBPROBLEMS_OPTION = "all"




def can_optimise_all_exclusion_subproblems(
    exclusion_count: int,
) -> bool:
    """
    Return whether automatic execution of all
    exclusion-generated subproblems is allowed.
    """

    return (
        0
        <= int(exclusion_count)
        <= MAX_EXCLUSIONS_FOR_ALL_SUBPROBLEMS
    )