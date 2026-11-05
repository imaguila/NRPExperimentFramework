# Location: src/domain/core/model.py

"""
Optimization Model Core Entity.

This module defines the central data container for optimization models.
The model is intentionally domain-agnostic, holding items, attributes,
relationships, evaluators, and metadata without enforcing domain-specific logic.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from src.domain.core.collection import SolutionSetCollection
from src.domain.core.definitions import (
    AttributeDefinition,
    ValueDependency,
)
from src.domain.core.item import DecisionItem

if TYPE_CHECKING:
    from src.domain.plugins.base_domain import DomainPlugin

def _json_serializer(obj: Any) -> Any:
    """
    Generic JSON serializer for non-primitive objects.

    Handles objects with a `to_dict()` method, dataclasses, sets,
    and ValueDependency instances. Falls back to string representation.
    """
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, ValueDependency):
        return obj.to_dict()
    return str(obj)


@dataclass
class OptimizationModel:
    """
    Generic optimization model container.

    This class is intentionally domain-agnostic, storing:
        - Decision items (variables)
        - Attribute definitions (metadata and aggregation rules)
        - Evaluators (stakeholders, developers, etc.)
        - Relationships (precedences, couplings, exclusions, value dependencies)
        - Solution collections (Pareto fronts, SOIs)
        - Raw input data and plugin reference

    It provides convenience methods for serialization, attribute introspection,
    and capacity calculations, but delegates domain-specific logic to plugins.

    Attributes:
        name: Human-readable model name.
        items: Mapping of item IDs to DecisionItem instances.
        attribute_definitions: Mapping of attribute names to their metadata.
        evaluators: Grouped evaluators (e.g., stakeholders, developers).
        relationships: Domain relationships (precedences, couplings, exclusions).
        value_dependencies: List of ValueDependency objects (synced with relationships).
        metadata: Arbitrary additional metadata.
        raw_data: Original raw data used to build the model (for provenance).
        plugin: Reference to the DomainPlugin that created this model.
        solutions_collection: Collection of stored Pareto fronts and SOIs.
    """
    name: str = "Optimization_Instance"
    items: Dict[str, DecisionItem] = field(default_factory=dict)
    attribute_definitions: Dict[str, AttributeDefinition] = field(default_factory=dict)
    evaluators: Dict[str, List[Any]] = field(default_factory=dict)
    relationships: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    value_dependencies: List[ValueDependency] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    plugin: Optional[Type[DomainPlugin]] = None
    solutions_collection: SolutionSetCollection = field(default_factory=SolutionSetCollection)

    def __post_init__(self) -> None:
        """
        Initialize derived fields and ensure consistency between data structures.

        - Sets a default name for the solutions collection if not already named.
        - Synchronizes value dependencies between relationships, raw_data, and the dedicated list.
        - Stores evaluators in metadata for backward compatibility.
        """
        if not self.solutions_collection.name or self.solutions_collection.name == "Optimization Study":
            self.solutions_collection.name = f"Results - {self.name}"

        if "value_dependencies" not in self.relationships:
            self.relationships["value_dependencies"] = []

        if "value_dependencies" not in self.raw_data:
            self.raw_data["value_dependencies"] = []

        if self.value_dependencies:
            vd_dicts = [
                vd.to_dict() if isinstance(vd, ValueDependency) else vd
                for vd in self.value_dependencies
            ]
            self.relationships["value_dependencies"] = vd_dicts
            self.raw_data["value_dependencies"] = vd_dicts

        if self.evaluators:
            self.metadata.setdefault("evaluators", self.evaluators)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the model to a generic serializable dictionary.

        If a plugin is attached and provides an `export_to_dict` method,
        it is used to produce a domain-specific representation.
        Otherwise, a generic structure is returned.

        Returns:
            A dictionary representation of the model.
        """
        if self.plugin and hasattr(self.plugin, "export_to_dict") and callable(self.plugin.export_to_dict):
            return self.plugin.export_to_dict(self)

        data = {
            "name": self.name,
            "num_items": len(self.items),
            "attribute_definitions": {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.attribute_definitions.items()
            },
            "items": [item.to_dict() for item in self.items.values()],
            "relationships": self.relationships,
            "evaluators": self.evaluators,
            "value_dependencies": [
                vd.to_dict() if isinstance(vd, ValueDependency) else vd
                for vd in self.value_dependencies
            ],
            "metadata": self.metadata,
        }
        return data

    def to_json(self) -> str:
        """
        Serialize the model to a JSON string.

        Returns:
            Pretty-printed JSON string.
        """
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=_json_serializer)

    def export_to_json(self) -> str:
        """Alias for to_json() for backward compatibility."""
        return self.to_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationModel":
        """
        Reconstruct an OptimizationModel from a dictionary.

        This is a generic parser. Domain-specific parsing (e.g., for NRP)
        should be handled by the plugin layer via GenericJSONLoader.

        Args:
            data: Dictionary containing the model data.

        Returns:
            An OptimizationModel instance.
        """
        attr_defs = {}
        for name, info in data.get("attribute_definitions", {}).items():
            if hasattr(info, "to_dict"):
                attr_defs[name] = info
            else:
                attr_defs[name] = AttributeDefinition.from_dict(name, info)

        raw_items = data.get("items") or data.get("requirements") or []
        items = {}
        for raw_item in raw_items:
            item = DecisionItem.from_dict(raw_item)
            items[item.id] = item

        relationships = data.get("relationships", {})
        if "value_dependencies" not in relationships:
            relationships["value_dependencies"] = data.get("value_dependencies", [])

        val_deps = data.get("value_dependencies") or relationships.get("value_dependencies", [])
        val_dep_objs = [ValueDependency.from_dict(vd) for vd in val_deps]

        evaluators = data.get("evaluators", {})

        return cls(
            name=str(data.get("name", "Optimization_Instance")),
            items=items,
            attribute_definitions=attr_defs,
            evaluators=evaluators,
            relationships=relationships,
            value_dependencies=val_dep_objs,
            raw_data=data,
        )

    def get_available_attributes(self) -> List[str]:
        """
        Return the union of all attribute names defined in the model.

        Combines attribute definitions and attributes present on individual items.

        Returns:
            Sorted list of attribute names.
        """
        attrs_from_defs = set(self.attribute_definitions.keys())
        attrs_from_items = set()
        for item in self.items.values():
            attrs_from_items.update(item.attributes.keys())
        return sorted(attrs_from_defs | attrs_from_items)

    def get_attribute_definition(self, attr_name: str) -> Optional[AttributeDefinition]:
        """
        Retrieve the definition of a specific attribute.

        Args:
            attr_name: Name of the attribute.

        Returns:
            The AttributeDefinition if found, otherwise None.
        """
        return self.attribute_definitions.get(attr_name)

    def get_total_capacity(self, attribute_name: str) -> float:
        """
        Compute the total capacity for a given attribute across all items.

        This sums the attribute values of all items in the model.

        Args:
            attribute_name: Name of the attribute.

        Returns:
            Total sum of the attribute across all items.
        """
        total = 0.0
        for item in self.items.values():
            total += item.get_attribute(attribute_name, 0.0)
        return total