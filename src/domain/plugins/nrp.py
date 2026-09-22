# Location: src/domain/plugins/nrp.py
"""
Next Release Problem (NRP) Domain Plugin Module.

This module provides the NRPPlugin class, which implements the DomainPlugin
interface for the Next Release Problem (NRP) domain. It handles attribute
schemas, data extraction, serialisation, synthetic instance generation,
preprocessing, enrichment, and stakeholder coverage analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from src.domain.core.definitions import AttributeDefinition
from src.domain.plugins.base_domain import DomainPlugin
from src.domain.plugins.nrp_schema import NRPAttributeSchema
from src.domain.plugins.nrp_serialization import export_to_dict, export_to_json
from src.domain.plugins.nrp_extraction import extract_items, extract_relationships, extract_evaluators
from src.domain.plugins.nrp_coverage import compute_stakeholder_coverage, build_stakeholder_requirement_matrix
from src.domain.plugins.nrp_enrichment import NRPEnricher
from src.domain.plugins import nrp_coupling, nrp_exclusion, nrp_value_dependencies

logger = logging.getLogger(__name__)


class NRPPlugin(DomainPlugin):
    """
    Domain plugin for the Next Release Problem (NRP).

    This plugin manages all NRP‑specific logic, including:
        - Attribute schema definition and retrieval.
        - Data extraction from raw JSON (items, relationships, evaluators).
        - Serialisation to and from the NRP canonical dictionary format.
        - Synthetic instance generation.
        - Preprocessing pipeline configuration (coupling, value dependencies, exclusion).
        - Enrichment indicators (productivity, effectiveness, etc.).
        - Stakeholder coverage analysis.
        - UI‑agnostic methods for labels and relationships.
        - Detection of decision variables and objective columns.

    It fully implements the DomainPlugin abstract interface, ensuring
    interoperability with the core framework.
    """

    # Plugin identification metadata
    NAME = "Next Release Problem"
    ID = "nrp"
    DESCRIPTION = "Software release planning requirement selection and sequencing optimisation."

    # UI‑agnostic labels and identifiers
    plugin_id: str = "nrp"
    display_name: str = "Next Release Problem (NRP)"
    item_label: str = "Requirement"
    item_label_plural: str = "Requirements"
    item_prefix: str = "R"

    SUPPORTED_AGGREGATIONS: List[str] = ["weighted_mean", "mean", "sum", "max", "min"]
    SUPPORTED_COUPLINGS: List[str] = ["sum", "max", "min"]
    DECISION_VAR_PREFIX: str = "req_"

    # --------------------------------------------------
    # Attribute Schema
    # --------------------------------------------------

    @classmethod
    def get_attribute_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return the NRP attribute schema."""
        return NRPAttributeSchema.get_attribute_schema()

    @classmethod
    def get_read_attributes(cls) -> Dict[str, AttributeDefinition]:
        """Return AttributeDefinition objects for all NRP attributes."""
        return NRPAttributeSchema.get_read_attributes()

    # --------------------------------------------------
    # Synthetic Instance Generation
    # --------------------------------------------------

    @classmethod
    def get_generation_config(cls) -> Dict[str, Any]:
        """Return the default configuration for synthetic NRP instance generation."""
        from src.domain.plugins.nrp_generation import get_nrp_generation_config
        return get_nrp_generation_config()

    @classmethod
    def generate_synthetic_instance(cls, **kwargs) -> Any:
        """Generate a synthetic NRP model instance."""
        from src.domain.plugins.nrp_generation import generate_nrp_instance
        return generate_nrp_instance(**kwargs)

    # --------------------------------------------------
    # Preprocessing Pipeline
    # --------------------------------------------------

    @classmethod
    def get_preprocessing_steps(cls) -> List[tuple]:
        """
        Return the list of preprocessing steps for the NRP domain.

        The steps are executed in order: coupling, value dependencies, exclusion.
        """
        return [
            ("coupling", nrp_coupling),
            ("value_dependencies", nrp_value_dependencies),
            ("exclusion", nrp_exclusion),
        ]

    # --------------------------------------------------
    # Data Extraction
    # --------------------------------------------------

    @classmethod
    def extract_items(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract DecisionItem objects from raw JSON data."""
        return extract_items(data)

    @classmethod
    def extract_relationships(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relationships (precedences, couplings, exclusions, value dependencies)."""
        return extract_relationships(data)

    @classmethod
    def extract_evaluators(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract evaluators (stakeholders and developers) from raw JSON data."""
        return extract_evaluators(data)

    # --------------------------------------------------
    # Serialisation / Export
    # --------------------------------------------------

    @classmethod
    def export_to_dict(cls, model) -> Dict[str, Any]:
        """Serialise an OptimizationModel to the NRP canonical dictionary format."""
        return export_to_dict(model)

    @classmethod
    def export_to_json(cls, model, indent: int = 2) -> str:
        """Serialise an OptimizationModel to a JSON string in the NRP format."""
        return export_to_json(model, indent)

    # --------------------------------------------------
    # Enrichment (delegated to NRPEnricher)
    # --------------------------------------------------

    @classmethod
    def get_calculated_indicators(cls) -> Dict[str, Any]:
        """Return the registry of available enrichment indicators."""
        return NRPEnricher.get_calculated_indicators()

    @classmethod
    def compute_indicators(cls, df: pd.DataFrame, indicators: Iterable[str], decision_var_prefix: str = "req_") -> pd.DataFrame:
        """Compute enrichment indicators on a DataFrame of solutions."""
        return NRPEnricher.compute_indicators(df, indicators, decision_var_prefix)

    @classmethod
    def get_enrichment_indicators(cls) -> Dict[str, Dict[str, Any]]:
        """Return enrichment indicator metadata (description and required attributes)."""
        indicators = NRPEnricher.get_calculated_indicators()
        return {
            name: {"description": ind.description, "required": ind.required_attributes}
            for name, ind in indicators.items()
        }

    @classmethod
    def compute_enrichment(cls, df: pd.DataFrame, indicators: List[str], decision_var_prefix: str = "req_") -> pd.DataFrame:
        """Compute enrichment indicators on a DataFrame of solutions."""
        return NRPEnricher.compute_indicators(df, indicators, decision_var_prefix)

    # --------------------------------------------------
    # Stakeholder Coverage
    # --------------------------------------------------

    @classmethod
    def compute_stakeholder_coverage(cls, df: pd.DataFrame, model) -> pd.DataFrame:
        """Compute stakeholder coverage scores for each solution."""
        return compute_stakeholder_coverage(df, model)

    @classmethod
    def build_stakeholder_requirement_matrix(cls, model) -> Optional[pd.DataFrame]:
        """Build the stakeholder‑requirement request matrix from the model."""
        return build_stakeholder_requirement_matrix(model)

    # --------------------------------------------------
    # UI‑Agnostic Methods
    # --------------------------------------------------

    @classmethod
    def get_display_name(cls) -> str:
        return "Next Release Problem (NRP)"

    @classmethod
    def get_item_label(cls) -> str:
        return "Requirement"

    @classmethod
    def get_item_label_plural(cls) -> str:
        return "Requirements"

    @classmethod
    def get_item_prefix(cls) -> str:
        return "R"

    @classmethod
    def get_relationship_label(cls, rel_type: str) -> str:
        labels = {
            "precedences": "Precedence",
            "couplings": "Coupling",
            "exclusions": "Exclusion",
            "value_dependencies": "Value Dependency"
        }
        return labels.get(rel_type, rel_type.capitalize())

    @classmethod
    def get_relationship_labels(cls) -> Dict[str, str]:
        return {
            "precedences": "Precedence",
            "couplings": "Coupling",
            "exclusions": "Exclusion",
            "value_dependencies": "Value Dependency"
        }

    @classmethod
    def get_supported_aggregations(cls) -> List[str]:
        return cls.SUPPORTED_AGGREGATIONS

    @classmethod
    def get_supported_couplings(cls) -> List[str]:
        return cls.SUPPORTED_COUPLINGS

    # --------------------------------------------------
    # Decision Variable and Objective Detection
    # --------------------------------------------------

    @classmethod
    def identify_decision_variables(cls, df: pd.DataFrame) -> List[str]:
        """
        Identify decision variable columns in a DataFrame.

        In NRP, decision variables are columns that start with `req_`.

        Args:
            df: DataFrame containing results.

        Returns:
            List of column names that represent decision variables.
        """
        if df is None or df.empty:
            return []
        return [col for col in df.columns if str(col).startswith(cls.DECISION_VAR_PREFIX)]

    @classmethod
    def identify_objective_columns(
        cls,
        df: pd.DataFrame,
        model: Optional[Any] = None,
        is_calculated: bool = False,
        configured_objectives: Optional[List[str]] = None,
        decision_vars: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Identify objective columns in a results DataFrame.

        For calculated Pareto fronts, uses the configured objectives.
        For loaded Pareto fronts (CSV), detects columns that are not decision
        variables but are present in the attribute schema.

        Args:
            df: DataFrame containing results.
            model: Optional model instance for additional attribute context.
            is_calculated: Whether the front was calculated (vs. loaded).
            configured_objectives: Objectives configured in the problem.
            decision_vars: Pre‑identified decision variable columns.

        Returns:
            List of column names that represent objectives.
        """
        if df is None or df.empty:
            return []

        # Case 1: Calculated Pareto front — use the configured objectives
        if is_calculated and configured_objectives:
            return [col for col in configured_objectives if col in df.columns]

        # Case 2: Loaded Pareto front (CSV) — detect from schema
        if decision_vars is None:
            decision_vars = cls.identify_decision_variables(df)
        dec_set = set(decision_vars)

        valid_attrs = set(cls.get_attribute_schema().keys())
        if model and hasattr(model, "get_available_attributes"):
            valid_attrs.update(model.get_available_attributes())

        normalized_valid = {str(attr).strip().lower() for attr in valid_attrs}

        return [
            col for col in df.columns
            if col not in dec_set and str(col).strip().lower() in normalized_valid
        ]