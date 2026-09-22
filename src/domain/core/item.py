# Location: src/domain/core/item.py
"""
Decision Item Core Entity.

This module defines the DecisionItem dataclass, which represents an individual
decision variable or requirement in the optimization domain. Each item holds
a set of attributes (scalars, multivalued lists, or aggregated values).
"""

from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class DecisionItem:
    """
    Represents an individual decision variable or requirement.

    The `attributes` dictionary contains all attribute values of the item:
        - Plain scalars (e.g., "effort": 4, "cost": 12.5)
        - Multivalued evaluator lists (e.g., "satisfaction": [1, 5], "risk": [3, 3])
        - Numeric aggregated values after preprocessing (e.g., "satisfaction": 3.86)

    Attributes:
        id: Unique identifier for the item.
        description: Human-readable description (optional).
        attributes: Dictionary of attribute names to their values.
    """

    id: str
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)

    def get_attribute(self, attr_name: str, default: float = 0.0) -> float:
        """
        Return the scalar numeric value of an attribute.

        Note: This method should **only** be used when the attribute has already
        been aggregated. For multivalued attributes that have not been aggregated,
        the preprocessing pipeline should handle the aggregation before
        optimisation.

        If the attribute is still a list of numbers, a warning is emitted and
        the first value is returned as a fallback.

        Args:
            attr_name: Name of the attribute to retrieve.
            default: Default value to return if the attribute is missing or
                cannot be converted to float.

        Returns:
            The numeric value of the attribute.
        """
        val = self.attributes.get(attr_name, default)

        if isinstance(val, (int, float)):
            return float(val)
        elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], (int, float)):
            # Fallback: if not aggregated, return the first value with a warning
            import warnings
            warnings.warn(
                f"Attribute '{attr_name}' for item '{self.id}' is still multivalued. "
                "Aggregation should happen before optimization."
            )
            return float(val[0])

        try:
            return float(val)
        except (ValueError, TypeError):
            return float(default)

    def set_attribute(self, attr_name: str, value: Any) -> None:
        """
        Set or update an attribute value for this item.

        Args:
            attr_name: Name of the attribute.
            value: New value for the attribute.
        """
        self.attributes[attr_name] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the item to a standard dictionary for JSON serialisation.

        Returns:
            Dictionary with 'id', 'description', and 'attributes' keys.
        """
        return {
            "id": self.id,
            "description": self.description,
            "attributes": self.attributes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionItem":
        """
        Create a DecisionItem instance from a dictionary.

        If the dictionary does not contain an 'attributes' key, all top-level
        keys except 'id' and 'description' are treated as attributes.

        Args:
            data: Dictionary with item data.

        Returns:
            A DecisionItem instance.
        """
        item_id = str(data.get("id", ""))
        desc = str(data.get("description", ""))
        attrs = data.get("attributes", {})

        # If properties are at the root level (outside 'attributes')
        if not attrs:
            attrs = {k: v for k, v in data.items() if k not in ("id", "description")}

        return cls(id=item_id, description=desc, attributes=attrs)

    def __repr__(self) -> str:
        return f"DecisionItem(id='{self.id}', attrs={list(self.attributes.keys())})"