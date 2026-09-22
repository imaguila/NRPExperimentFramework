# Location: src/domain/plugins/nrp_coupling.py

"""
Coupling Preprocessor using Disjoint-Set Union (DSU) for NRP.

This module merges coupled requirements, which must be selected together,
into unified decision items.

Precondition:
    Multivalued attributes must already have been resolved by
    GenericJSONLoader before this preprocessor is executed.

Responsibilities:
    - Identify connected coupling components using DSU.
    - Merge coupled decision items.
    - Aggregate scalar attributes using attribute-specific coupling rules.
    - Remap precedence relationships.
    - Remap exclusion relationships.
    - Remap value dependencies.
    - Remove the processed coupling relationships.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set

from src.domain.core.item import DecisionItem
from src.domain.core.model import OptimizationModel
from src.domain.plugins.nrp_utils import extract_pair


SUPPORTED_COUPLING_OPERATIONS = {
    "sum",
    "max",
    "min",
    "mean",
}


def _aggregate_values(
    values: List[Any],
    operation: str = "sum",
) -> float:
    """
    Aggregate scalar numeric values using a coupling operation.

    Multivalued attributes must have been resolved before this function
    is called.

    Args:
        values:
            Scalar values belonging to the items in a coupling cluster.

        operation:
            Aggregation operation. Supported values are:
            "sum", "max", "min", and "mean".

    Returns:
        Aggregated scalar value.

    Raises:
        ValueError:
            If a multivalued attribute reaches the coupling phase or if
            the requested coupling operation is not supported.

        TypeError:
            If a value cannot be converted to a floating-point number.
    """
    if not values:
        return 0.0

    for value in values:
        if isinstance(value, (list, tuple, dict)):
            raise ValueError(
                "Coupling received a multivalued attribute: "
                f"{value!r}. Multivalued attributes must be resolved "
                "before coupling preprocessing."
            )

    normalized_operation = str(operation).strip().lower()

    if normalized_operation not in SUPPORTED_COUPLING_OPERATIONS:
        raise ValueError(
            f"Unsupported coupling operation: "
            f"{normalized_operation!r}. Supported operations are: "
            f"{sorted(SUPPORTED_COUPLING_OPERATIONS)}."
        )

    numeric_values = [float(value) for value in values]

    if normalized_operation == "max":
        return float(max(numeric_values))

    if normalized_operation == "min":
        return float(min(numeric_values))

    if normalized_operation == "mean":
        return float(
            sum(numeric_values) / len(numeric_values)
        )

    return float(sum(numeric_values))




def run(
    model: OptimizationModel,
    custom_couplings: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> List[OptimizationModel]:
    """
    Apply coupling preprocessing to an optimization model.

    The function identifies connected components in the coupling graph
    using a Disjoint-Set Union structure. Each connected component is
    transformed into one DecisionItem.

    The attributes of the original items are combined using the coupling
    rules declared by the active domain plugin. Custom rules supplied by
    the caller override the plugin defaults.

    Precedences, exclusions, and value dependencies are remapped to the
    identifiers of the resulting unified items.

    Args:
        model:
            OptimizationModel to process.

        custom_couplings:
            Optional dictionary containing coupling-rule overrides,
            indexed by attribute name.

        **kwargs:
            Additional arguments accepted for compatibility with the
            generic preprocessing pipeline.

    Returns:
        A list containing one processed OptimizationModel.

        When the model contains no coupling relationships, the original
        model is returned unchanged inside the list.
    """
    couplings = model.relationships.get(
        "couplings",
        [],
    )

    if not couplings:
        return [model]

    # ============================================================
    # 1. OBTAIN ATTRIBUTE-SPECIFIC COUPLING RULES
    # ============================================================

    read_attributes = {}

    if (
        model.plugin is not None
        and hasattr(
            model.plugin,
            "get_read_attributes",
        )
    ):
        read_attributes = (
            model.plugin.get_read_attributes()
        )

    coupling_rules = {
        attribute_name: getattr(
            attribute_definition,
            "coupling_rule",
            "sum",
        )
        for (
            attribute_name,
            attribute_definition,
        ) in read_attributes.items()
    }

    if custom_couplings:
        coupling_rules.update(custom_couplings)

    # ============================================================
    # 2. BUILD COUPLING COMPONENTS USING DSU
    # ============================================================

    parent: Dict[str, str] = {}

    def find(item_id: str) -> str:
        """
        Find the representative identifier using path compression.
        """
        normalized_id = str(item_id).strip()

        if parent.setdefault(
            normalized_id,
            normalized_id,
        ) == normalized_id:
            return normalized_id

        parent[normalized_id] = find(
            parent[normalized_id]
        )

        return parent[normalized_id]

    def union(
        first_item_id: str,
        second_item_id: str,
    ) -> None:
        """
        Join two coupling components.
        """
        first_root = find(first_item_id)
        second_root = find(second_item_id)

        if first_root != second_root:
            parent[first_root] = second_root

    for coupling in couplings:
        first_item_id, second_item_id = (
            extract_pair(coupling)
        )

        if not first_item_id or not second_item_id:
            continue

        # Ignore references to unknown decision items.
        if (
            first_item_id not in model.items
            or second_item_id not in model.items
        ):
            continue

        union(
            first_item_id,
            second_item_id,
        )

    # ============================================================
    # 3. BUILD THE ITEM CLUSTERS
    # ============================================================

    clusters: Dict[str, Set[str]] = {}

    for item_id in model.items:
        normalized_id = str(item_id).strip()
        root_id = find(normalized_id)

        clusters.setdefault(
            root_id,
            set(),
        ).add(normalized_id)

    def get_unified_id(
        original_item_id: str,
    ) -> str:
        """
        Return the unified identifier for an original decision item.
        """
        normalized_id = str(
            original_item_id
        ).strip()

        root_id = find(normalized_id)

        if root_id in clusters:
            return "_".join(
                sorted(clusters[root_id])
            )

        return normalized_id

    id_map: Dict[str, str] = {
        str(item_id).strip(): get_unified_id(
            str(item_id)
        )
        for item_id in model.items
    }

    # ============================================================
    # 4. BUILD THE UNIFIED DECISION ITEMS
    # ============================================================

    new_items: Dict[str, DecisionItem] = {}

    for cluster_members in clusters.values():
        sorted_members = sorted(
            cluster_members
        )

        unified_id = "_".join(
            sorted_members
        )

        all_attribute_names: Set[str] = set()

        for member_id in sorted_members:
            if member_id not in model.items:
                continue

            all_attribute_names.update(
                model.items[
                    member_id
                ].attributes.keys()
            )

        combined_attributes: Dict[str, Any] = {}

        for attribute_name in all_attribute_names:
            attribute_values = [
                model.items[
                    member_id
                ].attributes[attribute_name]
                for member_id in sorted_members
                if (
                    member_id in model.items
                    and attribute_name
                    in model.items[
                        member_id
                    ].attributes
                )
            ]

            if not attribute_values:
                continue

            coupling_operation = (
                coupling_rules.get(
                    attribute_name,
                    "sum",
                )
            )

            combined_attributes[
                attribute_name
            ] = _aggregate_values(
                attribute_values,
                operation=coupling_operation,
            )

        if len(sorted_members) > 1:
            description = (
                "Unified item "
                f"({', '.join(sorted_members)})"
            )
        else:
            description = model.items[
                sorted_members[0]
            ].description

        new_items[unified_id] = DecisionItem(
            id=unified_id,
            description=description,
            attributes=combined_attributes,
        )

    # ============================================================
    # 5. REMAP PRECEDENCE RELATIONSHIPS
    # ============================================================

    new_precedences: List[Dict[str, str]] = []
    seen_precedences: Set[
        tuple[str, str]
    ] = set()

    for precedence in model.relationships.get(
        "precedences",
        [],
    ):
        source_id, target_id = extract_pair(
            precedence
        )

        if not source_id or not target_id:
            continue

        unified_source = id_map.get(
            source_id,
            source_id,
        )

        unified_target = id_map.get(
            target_id,
            target_id,
        )

        # A precedence internal to one unified item has already
        # been satisfied by the coupling operation.
        if unified_source == unified_target:
            continue

        precedence_key = (
            unified_source,
            unified_target,
        )

        if precedence_key in seen_precedences:
            continue

        seen_precedences.add(
            precedence_key
        )

        new_precedences.append(
            {
                "req1": unified_source,
                "req2": unified_target,
            }
        )

    # ============================================================
    # 6. REMAP EXCLUSION RELATIONSHIPS
    # ============================================================

    new_exclusions: List[Dict[str, str]] = []
    seen_exclusions: Set[
        tuple[str, str]
    ] = set()

    for exclusion in model.relationships.get(
        "exclusions",
        [],
    ):
        first_item_id, second_item_id = (
            extract_pair(exclusion)
        )

        if not first_item_id or not second_item_id:
            continue

        unified_first = id_map.get(
            first_item_id,
            first_item_id,
        )

        unified_second = id_map.get(
            second_item_id,
            second_item_id,
        )

        # Preserve the current behaviour: an exclusion whose
        # endpoints collapse into the same unified item is removed.
        if unified_first == unified_second:
            continue

        exclusion_key = tuple(
            sorted(
                (
                    unified_first,
                    unified_second,
                )
            )
        )

        if exclusion_key in seen_exclusions:
            continue

        seen_exclusions.add(
            exclusion_key
        )

        new_exclusions.append(
            {
                "req1": unified_first,
                "req2": unified_second,
            }
        )

    # ============================================================
    # 7. REMAP VALUE DEPENDENCIES
    # ============================================================

    raw_value_dependencies = (
        model.relationships.get(
            "value_dependencies",
            [],
        )
    )

    new_value_dependencies: List[Any] = []

    for dependency in raw_value_dependencies:
        if not isinstance(dependency, dict):
            new_value_dependencies.append(
                dependency
            )
            continue

        dependency_copy = copy.deepcopy(
            dependency
        )

        raw_requirement_ids = (
            dependency_copy.get(
                "reqs",
                [],
            )
        )

        mapped_requirement_ids = [
            id_map.get(
                str(requirement_id).strip(),
                str(requirement_id).strip(),
            )
            for requirement_id
            in raw_requirement_ids
        ]

        # Remove duplicate identifiers while preserving order.
        dependency_copy["reqs"] = list(
            dict.fromkeys(
                mapped_requirement_ids
            )
        )

        new_value_dependencies.append(
            dependency_copy
        )

    # ============================================================
    # 8. BUILD RESULTING RELATIONSHIPS
    # ============================================================

    new_relationships = copy.deepcopy(
        model.relationships
    )

    new_relationships[
        "precedences"
    ] = new_precedences

    # Couplings have been absorbed into the unified items.
    new_relationships[
        "couplings"
    ] = []

    new_relationships[
        "exclusions"
    ] = new_exclusions

    new_relationships[
        "value_dependencies"
    ] = new_value_dependencies

    # ============================================================
    # 9. BUILD RESULTING MODEL
    # ============================================================

    processed_model = OptimizationModel(
        name=model.name,
        items=new_items,
        attribute_definitions=copy.deepcopy(
            getattr(
                model,
                "attribute_definitions",
                {},
            )
        ),
        evaluators=copy.deepcopy(
            getattr(
                model,
                "evaluators",
                {},
            )
        ),
        relationships=new_relationships,
        metadata=copy.deepcopy(
            getattr(
                model,
                "metadata",
                {},
            )
        ),
        raw_data=copy.deepcopy(
            getattr(
                model,
                "raw_data",
                {},
            )
        ),
        plugin=getattr(
            model,
            "plugin",
            None,
        ),
    )

    return [processed_model]