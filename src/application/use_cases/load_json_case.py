# src/application/use_cases/load_json_case.py
"""
Use cases for inspecting, loading, and preprocessing JSON problem instances.

This module is independent of Streamlit. It can be used from:

- The Streamlit interface.
- Jupyter notebooks.
- Automated tests.
- Command-line clients.
- Future API clients.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.application.dto import (
    JsonCaseInspection,
    LoadJsonCaseResult,
)
from src.domain.core.loader import GenericJSONLoader
from src.domain.core.pipeline import PreprocessingEngine
from src.domain.plugins.base_domain import DomainPlugin


def parse_json_object(
    json_input: str,
) -> Dict[str, Any]:
    """
    Parse a JSON string and validate that its root is an object.

    Args:
        json_input:
            JSON document represented as text.

    Returns:
        Parsed JSON dictionary.

    Raises:
        json.JSONDecodeError:
            If the input is not valid JSON.

        ValueError:
            If the root JSON element is not an object.
    """
    parsed_data = json.loads(json_input)

    if not isinstance(parsed_data, dict):
        raise ValueError(
            "The root element of the JSON instance must be an object."
        )

    return parsed_data


def inspect_json_case(
    json_input: str,
    plugin: DomainPlugin,
) -> JsonCaseInspection:
    """
    Inspect a JSON instance using its declared default aggregation rules.

    The inspected model allows clients to discover:

    - Attribute definitions.
    - Multivalued attributes.
    - Default aggregation rules.
    - Available model attributes.
    - Coupling relationships.
    - Default coupling rules.

    Args:
        json_input:
            JSON document represented as text.

        plugin:
            Active domain plugin.

    Returns:
        JsonCaseInspection containing the parsed data and inspected model.
    """
    raw_data = parse_json_object(json_input)

    inspected_model = GenericJSONLoader.load(
        json_input=json_input,
        plugin=plugin,
    )

    return JsonCaseInspection(
        raw_data=raw_data,
        inspected_model=inspected_model,
    )


def load_json_case(
    json_input: str,
    plugin: DomainPlugin,
    aggregation_overrides: Optional[Dict[str, str]] = None,
    coupling_overrides: Optional[Dict[str, str]] = None,
) -> LoadJsonCaseResult:
    """
    Load and preprocess a JSON optimisation instance.

    Processing order:

    1. Parse and validate the JSON document.
    2. Extract and materialise the model through the active plugin.
    3. Resolve multivalued attributes using declared or overridden rules.
    4. Execute coupling preprocessing.
    5. Execute value-dependency preprocessing.
    6. Execute exclusion branching.

    Args:
        json_input:
            JSON document represented as text.

        plugin:
            Active domain plugin.

        aggregation_overrides:
            Optional custom aggregation rules indexed by attribute name.

        coupling_overrides:
            Optional custom coupling rules indexed by attribute name.

    Returns:
        LoadJsonCaseResult containing the original JSON data, the loaded
        base model, and every model variant produced by preprocessing.

    Raises:
        json.JSONDecodeError:
            If the JSON document is invalid.

        ValueError:
            If the JSON structure, model relationships, aggregation rules,
            or preprocessing configuration are invalid.

        RuntimeError:
            If the preprocessing pipeline produces no model variants.
    """
    raw_data = parse_json_object(json_input)

    base_model = GenericJSONLoader.load(
        json_input=json_input,
        plugin=plugin,
        attr_aggregations=aggregation_overrides or None,
    )

    pipeline_options: Dict[str, Dict[str, Any]] = {}

    if coupling_overrides:
        pipeline_options["coupling"] = {
            "custom_couplings": dict(coupling_overrides)
        }

    processed_models = PreprocessingEngine.process(
        initial_model=base_model,
        pipeline_options=pipeline_options,
    )

    if not processed_models:
        raise RuntimeError(
            "The preprocessing pipeline did not produce any model variant."
        )

    return LoadJsonCaseResult(
        raw_data=raw_data,
        base_model=base_model,
        processed_models=list(processed_models),
    )