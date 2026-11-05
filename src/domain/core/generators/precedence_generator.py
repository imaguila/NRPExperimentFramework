# Location: src/domain/core/generators/precedence_generator.py
"""
Generic Directed Acyclic Graph (DAG) Precedence Generator.
"""

import random
import copy
from typing import List, Dict
from src.domain.core.model import OptimizationModel


def generate_random_dag(
    item_ids: List[str],
    density: float = 0.2,
) -> List[Dict[str, str]]:
    """
    Generate a valid set of precedence relationships (DAG),
    ensuring that no dependency cycles are introduced.

    Under the NRP-Exp convention:

        req1 requires req2

    Therefore, a generated relationship:

        {"req1": "R1", "req2": "R2"}

    means that selecting R1 requires R2.

    Args:
        item_ids:
            List of item identifiers.

        density:
            Probability (0.0-1.0) that a precedence
            relationship is generated between two items.

    Returns:
        List of precedence relationships.
    """

    nodes = [str(x) for x in item_ids]
    precedences = []
    n = len(nodes)

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < density:
                # Under the NRP-Exp convention:
                #     nodes[i] requires nodes[j]
                # represented as:
                #     {"req1": nodes[i],
                #      "req2": nodes[j]}
                #
                precedences.append(
                    {
                        "req1": nodes[i],
                        "req2": nodes[j],
                    }
                )

    return precedences


def inject_precedences_into_model(
    model: OptimizationModel,
    density: float = 0.2,
) -> OptimizationModel:
    """
    Generate precedence relationships for the model items
    and inject them into:

        model.relationships["precedences"]

    without modifying the original model.

    Args:
        model:
            Source model.

        density:
            Precedence density.

    Returns:
        A new model instance containing the generated
        precedence relationships.
    """

    # Create a deep copy to preserve the original model.
    new_model = copy.deepcopy(model)

    item_ids = list(new_model.items.keys())
    new_precedences = generate_random_dag(
        item_ids,
        density=density,
    )

    if not isinstance(new_model.relationships, dict):
        new_model.relationships = {}

    new_model.relationships["precedences"] = new_precedences

    return new_model