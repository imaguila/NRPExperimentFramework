# Location: src/domain/core/definitions.py
"""
Domain Core Definitions.

This module provides dataclasses for the core domain concepts:
- Evaluator (stakeholder, developer, or any preference‑ranking entity)
- AttributeDefinition (metadata and rules for problem attributes)
- ValueDependency (non‑linear value/cost interactions between items)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Evaluator:
    """
    Represents a preference evaluator (e.g., client, developer).

    Attributes:
        id: Unique identifier for the evaluator.
        weight: Importance weight of this evaluator (used in weighted aggregation).
        description: Human‑readable description.
        group: Group name (e.g., 'stakeholders', 'developers', 'default').
    """
    id: str
    weight: float = 1.0
    description: str = ""
    group: str = "default"  # 'stakeholders', 'developers', etc.

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the evaluator to a dictionary.

        Returns:
            Dictionary with evaluator data.
        """
        return {
            "id": self.id,
            "weight": self.weight,
            "description": self.description,
            "group": self.group
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evaluator":
        """
        Reconstruct an Evaluator from a dictionary.

        Args:
            data: Dictionary containing evaluator data.

        Returns:
            An Evaluator instance.
        """
        return cls(
            id=str(data.get("id", "")),
            weight=float(data.get("weight", 1.0)),
            description=str(data.get("description", "")),
            group=str(data.get("group", "default"))
        )


@dataclass
class AttributeDefinition:
    """
    Metadata and rules for a problem attribute.

    Attributes:
        name: Attribute name.
        type: Data type – 'scalar' or 'multivalued'.
        description: Human‑readable description.
        evaluator_group: Group of evaluators relevant for this attribute
            (e.g., 'stakeholders' for satisfaction).
        aggregation_rule: How multivalued attributes are aggregated into a
            scalar ('sum', 'mean', 'weighted_mean', 'max', 'min').
        coupling_rule: How coupled items are combined for this attribute
            ('sum', 'max', 'min').
        evaluation_rule: How a solution's total value for this attribute is
            computed from selected items ('sum', 'max', 'min').
    """
    name: str
    type: str = "scalar"  # 'scalar' or 'multivalued'
    description: str = ""
    evaluator_group: Optional[str] = None
    aggregation_rule: str = "weighted_mean"  # sum, mean, weighted_mean, max, min
    coupling_rule: str = "sum"               # sum, max, min
    evaluation_rule: str = "sum"             # sum, max, min

    @property
    def is_multivalued(self) -> bool:
        """Return True if the attribute is multivalued."""
        return self.type == "multivalued"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the attribute definition to a dictionary.

        Returns:
            Dictionary with attribute metadata.
        """
        data = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "coupling_rule": self.coupling_rule,
            "aggregation_rule": self.aggregation_rule,
            "evaluation_rule": self.evaluation_rule
        }
        if self.evaluator_group:
            data["evaluator_group"] = self.evaluator_group
        return data

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "AttributeDefinition":
        """
        Reconstruct an AttributeDefinition from a dictionary.

        Args:
            name: Attribute name (used as the key).
            data: Dictionary containing attribute metadata.

        Returns:
            An AttributeDefinition instance.
        """
        return cls(
            name=name,
            type=str(data.get("type", "scalar")),
            description=str(data.get("description", "")),
            evaluator_group=data.get("evaluator_group"),
            aggregation_rule=str(data.get("aggregation_rule", "weighted_mean")),
            coupling_rule=str(data.get("coupling_rule", "sum")),
            evaluation_rule=str(data.get("evaluation_rule", "sum"))
        )


@dataclass
class ValueDependency:
    """
    Represents a non‑linear value/cost interaction between items.

    For example, a synergy where including two items together reduces the
    total effort (multiplier) or adds a bonus to satisfaction (delta).

    Attributes:
        attribute: The attribute affected by this interaction.
        reqs: List of item IDs involved in the interaction.
        effect: Type of effect – 'multiplier' or 'delta'.
        factor: Numerical factor for the effect (multiplier value or delta amount).
        description: Human‑readable description of the interaction.
    """
    attribute: str
    reqs: List[str]
    effect: str          # 'multiplier', 'delta'
    factor: float
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the value dependency to a dictionary.

        Returns:
            Dictionary with dependency data.
        """
        return {
            "attribute": self.attribute,
            "reqs": self.reqs,
            "effect": self.effect,
            "factor": self.factor,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValueDependency":
        """
        Reconstruct a ValueDependency from a dictionary.

        Args:
            data: Dictionary containing dependency data.

        Returns:
            A ValueDependency instance.
        """
        return cls(
            attribute=str(data.get("attribute", "")),
            reqs=list(data.get("reqs", [])),
            effect=str(data.get("effect", "delta")),
            factor=float(data.get("factor", 1.0)),
            description=str(data.get("description", ""))
        )