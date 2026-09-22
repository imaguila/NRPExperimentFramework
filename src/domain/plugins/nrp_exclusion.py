# Location: src/domain/plugins/nrp_exclusion.py
"""
Exclusion Preprocessor for NRP.

This module handles mutual exclusion constraints by generating all maximal
valid branches. For each exclusion pair (A, B), it creates two branches:
one where A is removed (along with everything that structurally requires
it, and its coupled items) and one where B is removed. This process is
applied recursively until no exclusions remain.

Removal semantics
------------------
Precedences are directional: ``(from=X, to=Y)`` means "selecting X requires
Y" (X depends on Y). When an item ``i`` is removed:

- Every item that **requires** ``i`` (i.e. every ``X`` with a precedence
  ``X -> i``, directly or transitively) can never satisfy that requirement
  again, since ``i`` no longer exists as an option. These items are
  therefore also removed, cascading through the *reverse* precedence graph.
- Items that ``i`` itself requires (i.e. every ``Y`` with ``i -> Y``) are
  **not** removed solely because ``i`` is gone: they may still be selected
  independently, or be required by other retained items.
- Items coupled with anything being removed are removed as well (coupling
  forces joint selection), and the cascade continues from them too (an
  item that required a now-removed coupled item must also go).

This mirrors the paper's ``Dep(i)`` definition (Eq. 9): ``Dep(i)`` is the
set of ``i`` plus every remaining item whose implication condition
requires ``i``, directly or transitively — i.e. predecessors, not
successors, of ``i`` in the precedence graph.
"""

from __future__ import annotations

import copy
from typing import List, Dict, Set, Any

from src.domain.core.item import DecisionItem
from src.domain.plugins.nrp_utils import extract_pair


def _get_coupled_items(model, item_id: str) -> Set[str]:
    """
    Return the set of items that are coupled with the given item.

    Couplings are defined in the model's relationships under the 'couplings' key.

    Args:
        model: The OptimizationModel instance.
        item_id: ID of the item to check.

    Returns:
        Set of coupled item IDs.
    """
    couplings = model.relationships.get("couplings", [])
    coupled = set()
    for c in couplings:
        r1, r2 = extract_pair(c)
        if r1 == item_id and r2 in model.items:
            coupled.add(r2)
        elif r2 == item_id and r1 in model.items:
            coupled.add(r1)
    return coupled


def _build_reverse_precedence_graph(precedences) -> Dict[str, Set[str]]:
    """
    Build the reverse precedence graph: for each ``(src, tgt)`` pair
    ("selecting src requires tgt"), record ``reverse[tgt] = {..., src}``.

    This lets us answer, for a given item ``tgt``, "which items directly
    require me?" — the set we must cascade-remove when ``tgt`` disappears.

    Args:
        precedences: Iterable of relationship entries (dict/tuple/list)
            understood by ``extract_pair``.

    Returns:
        Mapping from item id to the set of items that directly require it.
    """
    reverse_graph: Dict[str, Set[str]] = {}
    for rule in precedences:
        src, tgt = extract_pair(rule)
        if src and tgt:
            reverse_graph.setdefault(tgt, set()).add(src)
    return reverse_graph


def remove_item_and_dependents(model, item_id_to_remove: str):
    """
    Create a new model where the specified item, every remaining item that
    structurally requires it (directly or transitively, via the reverse
    precedence graph), and every item coupled with any of those, are
    removed.

    Args:
        model: The original OptimizationModel.
        item_id_to_remove: ID of the item to delete.

    Returns:
        A new OptimizationModel with the item and its dependents removed.
    """
    from src.domain.core.model import OptimizationModel  # Local import to avoid circular dependency

    new_items = copy.deepcopy(model.items)
    new_relationships = copy.deepcopy(model.relationships)

    # Reverse precedence graph: item -> items that require it (predecessors).
    precedences = new_relationships.get("precedences", [])
    reverse_graph = _build_reverse_precedence_graph(precedences)

    # BFS to find all items that must be removed: the seed item, every item
    # that (transitively) requires it, and everything coupled along the way.
    to_remove: Set[str] = set()
    queue = [str(item_id_to_remove).strip()]
    while queue:
        current = queue.pop(0)
        if current in to_remove:
            continue
        to_remove.add(current)

        # Items that require 'current' can no longer satisfy that
        # requirement once 'current' is gone — they must be removed too.
        for requirer in reverse_graph.get(current, set()):
            if requirer not in to_remove:
                queue.append(requirer)

        # Items coupled with 'current' must be removed together with it,
        # and the cascade continues from them (something may in turn
        # require the coupled item).
        coupled = _get_coupled_items(model, current)
        for c in coupled:
            if c not in to_remove:
                queue.append(c)

    # Remove the identified items
    for item_id in to_remove:
        new_items.pop(item_id, None)

    # Clean up precedences: keep only those whose endpoints are not removed.
    # (No dangling precedence can remain: any item that required a removed
    # 'to' endpoint was itself added to `to_remove` above via the reverse
    # graph, so both endpoints of a surviving precedence are guaranteed to
    # still exist.)
    new_precedences = []
    for rule in new_relationships.get("precedences", []):
        src, tgt = extract_pair(rule)
        if src not in to_remove and tgt not in to_remove:
            new_precedences.append({"req1": src, "req2": tgt})

    # Clean up exclusions
    new_exclusions = []
    for rule in new_relationships.get("exclusions", []):
        e1, e2 = extract_pair(rule)
        if e1 not in to_remove and e2 not in to_remove:
            new_exclusions.append({"req1": e1, "req2": e2})

    # Clean up value dependencies: keep only those whose reqs are not removed
    new_val_deps = []
    for dep in new_relationships.get("value_dependencies", []):
        if isinstance(dep, dict):
            reqs = [str(r) for r in dep.get("reqs", []) if str(r) not in to_remove]
            if reqs:
                dep_copy = copy.deepcopy(dep)
                dep_copy["reqs"] = reqs
                new_val_deps.append(dep_copy)
        else:
            new_val_deps.append(dep)

    new_relationships["precedences"] = new_precedences
    new_relationships["exclusions"] = new_exclusions
    new_relationships["value_dependencies"] = new_val_deps

    # Build and return the new model
    return OptimizationModel(
        name=model.name,
        items=new_items,
        attribute_definitions=copy.deepcopy(
            getattr(model, "attribute_definitions", {})
        ),
        evaluators=copy.deepcopy(
            getattr(model, "evaluators", {})
        ),
        relationships=new_relationships,
        metadata=copy.deepcopy(
            getattr(model, "metadata", {})
        ),
        raw_data=copy.deepcopy(
            getattr(model, "raw_data", {})
        ),
        plugin=getattr(model, "plugin", None),
    )


def branch_recursive(current_model, exclusions: List[Any], current_path: str = "Base") -> List:
    """
    Recursively generate all maximal valid branches by resolving exclusions.

    For each exclusion pair (A, B), two branches are generated:
    - One where A is removed (with everything that requires it/couplings).
    - One where B is removed.
    This process continues for the remaining exclusions in each branch.

    Args:
        current_model: The current model to branch from.
        exclusions: List of remaining exclusion pairs to resolve.
        current_path: String describing the path taken (for naming).

    Returns:
        A list of models, each representing a maximal valid variant.
    """
    if not exclusions:
        current_model.branch_name = current_path
        return [current_model]

    first_excl, *remaining_exclusions = exclusions
    r1, r2 = extract_pair(first_excl)

    # If either item is already absent, this exclusion is already satisfied
    if not r1 or not r2 or r1 not in current_model.items or r2 not in current_model.items:
        return branch_recursive(current_model, remaining_exclusions, current_path)

    # Branch A: remove r1
    model_a = remove_item_and_dependents(current_model, r1)
    path_a = f"{current_path} ➔ Excl({r1})" if current_path != "Base" else f"Excl({r1})"
    branches_a = branch_recursive(model_a, remaining_exclusions, path_a)

    # Branch B: remove r2
    model_b = remove_item_and_dependents(current_model, r2)
    path_b = f"{current_path} ➔ Excl({r2})" if current_path != "Base" else f"Excl({r2})"
    branches_b = branch_recursive(model_b, remaining_exclusions, path_b)

    return branches_a + branches_b


def run(model, **kwargs) -> List:
    """
    Entry point for the exclusion preprocessor.

    If the model contains exclusions, it generates all maximal valid branches
    by recursively resolving them. If no exclusions are present, it returns
    the original model with branch_name set to 'Base'.

    Args:
        model: The OptimizationModel to process.
        **kwargs: Additional arguments (unused).

    Returns:
        A list of OptimizationModel instances (one or more branches).
    """
    exclusions = model.relationships.get("exclusions", [])
    if not exclusions:
        model.branch_name = getattr(model, "branch_name", "Base")
        return [model]
    return branch_recursive(model, exclusions)
