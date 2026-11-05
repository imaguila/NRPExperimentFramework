# Location: src/domain/plugins/nrp_schema.py
"""
NRP Attribute Schema definitions.

This module centralises the attribute schema for the Next Release Problem (NRP)
domain, providing the canonical definition of all available attributes, their
types, aggregation rules, coupling strategies, and default distributions.
"""

from typing import Any, Dict

from src.domain.core.definitions import AttributeDefinition


class NRPAttributeSchema:
    """
    Centralised attribute schema for the NRP domain.

    This class provides static methods to retrieve the complete attribute
    schema and to build AttributeDefinition objects from it.
    """

    @staticmethod
    def get_attribute_schema() -> Dict[str, Dict[str, Any]]:
        """
        Return the complete attribute schema for the NRP domain.

        The schema defines each attribute's type (scalar or multivalued),
        display label, description, coupling rule, evaluation rule, whether
        it is integer, and its default distribution for synthetic generation.

        Returns:
            Dictionary mapping attribute names to their configuration.
        """
        return {
            "effort": {
                "type": "scalar",
                "label": "Effort / Development Cost",
                "description": "Total developer effort required",
                "coupling_rule": "sum",
                "evaluation_rule": "sum",
                "is_integer": True,
                "default_dist": "fibonacci"
            },
            "cost": {
                "type": "scalar",
                "label": "Financial Cost",
                "description": "Direct financial cost",
                "coupling_rule": "sum",
                "evaluation_rule": "sum",
                "is_integer": True,
                "default_dist": "uniform"
            },
            "time": {
                "type": "scalar",
                "label": "Time / Duration",
                "description": "Execution or delivery time",
                "coupling_rule": "max",
                "evaluation_rule": "max",
                "is_integer": True,
                "default_dist": "uniform"
            },
            "prevalence": {
                "type": "scalar",
                "label": "Prevalence / Usage",
                "description": "Usage frequency or prevalence score",
                "coupling_rule": "sum",
                "evaluation_rule": "sum",
                "is_integer": True,
                "default_dist": "uniform"
            },
            "instability": {
                "type": "scalar",
                "label": "Instability",
                "description": "Requirement volatility or instability index",
                "coupling_rule": "max",
                "evaluation_rule": "max",
                "is_integer": True,
                "default_dist": "uniform"
            },
            "satisfaction": {
                "type": "multivalued",
                "label": "Client Satisfaction",
                "description": "Client satisfaction gain if requirement is included",
                "coupling_rule": "sum",
                "evaluation_rule": "sum",
                "aggregation_rule": "weighted_mean",
                "evaluator_group": "stakeholders",
                "is_integer": True,
                "default_dist": "uniform"
            },
            "dissatisfaction": {
                "type": "multivalued",
                "label": "Client Dissatisfaction",
                "description": "Client dissatisfaction penalty if requirement is excluded",
                "coupling_rule": "sum",
                "evaluation_rule": "sum",
                "aggregation_rule": "weighted_mean",
                "evaluator_group": "stakeholders",
                "is_integer": True,
                "default_dist": "uniform"
            },
            "risk": {
                "type": "multivalued",
                "label": "Technical Risk",
                "description": "Technical risk level evaluated by developers",
                "coupling_rule": "max",
                "evaluation_rule": "max",
                "aggregation_rule": "weighted_mean",
                "evaluator_group": "developers",
                "is_integer": True,
                "default_dist": "uniform"
            }
        }

    @staticmethod
    def get_read_attributes() -> Dict[str, AttributeDefinition]:
        """
        Build AttributeDefinition objects from the schema.

        This method converts the raw schema dictionary into a dictionary of
        AttributeDefinition instances, which are used by the loader and
        preprocessors to apply aggregation and coupling rules.

        Returns:
            Dictionary mapping attribute names to AttributeDefinition objects.
        """
        schema = NRPAttributeSchema.get_attribute_schema()
        read_attrs = {}
        for attr_id, info in schema.items():
            aggregation_rule = info.get("aggregation_rule")
            if aggregation_rule is None:
                aggregation_rule = "weighted_mean"
            read_attrs[attr_id] = AttributeDefinition(
                name=attr_id,
                type=info["type"],
                description=info["description"],
                evaluator_group=info.get("evaluator_group"),
                coupling_rule=info["coupling_rule"],
                aggregation_rule=aggregation_rule,
                evaluation_rule=info.get("evaluation_rule", "sum")
            )
        return read_attrs