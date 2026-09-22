# Location: src/domain/core/loader.py
"""
JSON Loader for Optimization Models.

This module provides a generic loader that delegates all domain‑specific
extraction logic to the plugin, ensuring full domain agnosticism.
"""

import json
from typing import Any, Dict, List, Optional, Union

from src.domain.core.item import DecisionItem
from src.domain.core.model import OptimizationModel
from src.domain.core.definitions import AttributeDefinition, Evaluator


class GenericJSONLoader:
    """
    Infrastructure loader that delegates domain extraction strictly to the plugin.

    The loader handles input normalisation (JSON string, bytes, or dict),
    invokes plugin methods to extract items, relationships, and evaluators,
    builds the OptimizationModel, and finally resolves multivalued attributes
    using the plugin's aggregation rules.
    """

    @classmethod
    def load(
        cls,
        json_input: Union[str, Dict[str, Any], bytes],
        plugin: Any,
        attr_aggregations: Optional[Dict[str, str]] = None
    ) -> OptimizationModel:
        """
        Load an OptimizationModel from JSON input using the provided plugin.

        Args:
            json_input: JSON string, bytes, or dictionary.
            plugin: Domain plugin instance (must implement extract_items,
                extract_relationships, extract_evaluators, and clean_attribute_name).
            attr_aggregations: Optional override of aggregation rules for
                specific attributes (attribute name -> aggregation rule).

        Returns:
            An OptimizationModel instance with all data extracted and aggregated.

        Raises:
            ValueError: If no plugin is provided.
        """
        if not plugin:
            raise ValueError("[GenericJSONLoader] An explicit 'plugin' is required to load the model.")

        # 1. Normalise input
        if isinstance(json_input, bytes):
            data = json.loads(json_input.decode("utf-8"))
        elif isinstance(json_input, str):
            data = json.loads(json_input)
        else:
            data = json_input

        # 2. Delegate domain extraction to the plugin
        items = plugin.extract_items(data)
        relationships = plugin.extract_relationships(data)
        evaluators = plugin.extract_evaluators(data)

        # Convert raw attribute definitions to objects
        attribute_definitions = {}
        raw_attr_defs = data.get("attribute_definitions", {})
        if raw_attr_defs:
            for name, info in raw_attr_defs.items():
                attribute_definitions[name] = AttributeDefinition.from_dict(name, info)

        # 3. Build the model
        model = OptimizationModel(
            name=data.get("name", "Optimization_Instance"),
            items=items,
            attribute_definitions=attribute_definitions,
            relationships=relationships,
            evaluators=evaluators,
            plugin=plugin,
            raw_data=data
        )

        # 4. Resolve multivalued attributes (mathematical aggregation)
        cls._resolve_multivalued(model, attr_aggregations)
        return model

    @classmethod
    def _resolve_multivalued(
        cls,
        model: OptimizationModel,
        custom_aggregations: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Resolve multivalued attributes using the plugin's aggregation rules.

        For each item, multivalued attributes (lists or dicts) are aggregated
        into a single scalar according to the rule defined in the attribute
        definition or plugin's read_attributes. If the rule is 'weighted_mean'
        and the attribute has an evaluator_group, the corresponding evaluator
        weights are used.

        Args:
            model: The model whose items' attributes will be resolved.
            custom_aggregations: Optional per-attribute aggregation rule overrides.
        """
        # 1. Obtain aggregation rules from the plugin
        read_attrs = model.plugin.get_read_attributes() if hasattr(model.plugin, "get_read_attributes") else {}
        agg_rules = {k: v.aggregation_rule for k, v in read_attrs.items() if hasattr(v, "aggregation_rule")}

        # Override with definitions from the JSON if they exist
        if model.attribute_definitions:
            for attr_name, attr_def in model.attribute_definitions.items():
                if hasattr(attr_def, "aggregation_rule"):
                    agg_rules[attr_name] = attr_def.aggregation_rule

        if custom_aggregations:
            agg_rules.update(custom_aggregations)

        # 2. Extract evaluator weights by group (now using Evaluator objects)
        evaluator_weights = {}
        for group, evals in model.evaluators.items():
            if evals and isinstance(evals[0], Evaluator):
                evaluator_weights[group] = [e.weight for e in evals]
            elif evals and isinstance(evals[0], dict):
                evaluator_weights[group] = [float(e.get("weight", 1.0)) for e in evals]
            else:
                evaluator_weights[group] = []

        # 3. Process each item
        for item in model.items.values():
            resolved_attributes: Dict[str, Any] = {}
            for attr_name, attr_val in item.attributes.items():
                clean_name = model.plugin.clean_attribute_name(attr_name)
                attr_def = model.attribute_definitions.get(clean_name)

                if isinstance(attr_val, (list, dict)):
                    # Normalise to a list of values
                    if isinstance(attr_val, dict):
                        values_list = list(attr_val.values())
                    else:
                        values_list = attr_val

                    # Keep only numeric values
                    numeric_vals = [float(v) for v in values_list if isinstance(v, (int, float))]
                    if not numeric_vals:
                        continue

                    # Determine aggregation operator
                    op = agg_rules.get(clean_name, "mean")

                    if op == "weighted_mean" and attr_def and attr_def.evaluator_group:
                        group = attr_def.evaluator_group
                        weights = evaluator_weights.get(group, [])
                        if weights and len(weights) == len(numeric_vals):
                            total_weight = sum(weights)
                            if total_weight > 0:
                                weighted_sum = sum(v * w for v, w in zip(numeric_vals, weights))
                                resolved_attributes[clean_name] = float(weighted_sum / total_weight)
                            else:
                                resolved_attributes[clean_name] = float(sum(numeric_vals) / len(numeric_vals))
                        else:
                            # Fallback to simple mean
                            resolved_attributes[clean_name] = float(sum(numeric_vals) / len(numeric_vals))
                    elif op == "sum":
                        resolved_attributes[clean_name] = float(sum(numeric_vals))
                    elif op == "max":
                        resolved_attributes[clean_name] = float(max(numeric_vals))
                    elif op == "min":
                        resolved_attributes[clean_name] = float(min(numeric_vals))
                    else:  # default to mean
                        resolved_attributes[clean_name] = float(sum(numeric_vals) / len(numeric_vals))
                else:
                    # Scalar attribute – keep as is
                    resolved_attributes[clean_name] = attr_val

            item.attributes = resolved_attributes
            