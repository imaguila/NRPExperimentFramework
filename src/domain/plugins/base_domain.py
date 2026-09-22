# Location: src/domain/plugins/base_domain.py
"""
Base Abstract Interface for Domain Plugins.

This module defines the abstract base class that all domain plugins must implement.
It provides the contract for domain-specific logic such as attribute schemas,
data extraction, serialization, enrichment, and synthetic instance generation.
"""

from abc import ABC, abstractmethod
import json
import pandas as pd
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.core.model import OptimizationModel
    from src.domain.core.definitions import AttributeDefinition


class DomainPlugin(ABC):
    """
    Abstract base interface for optimisation domain plugins.

    All domain-specific plugins (e.g., NRP, Portfolio) must inherit from this class
    and implement the abstract methods. This ensures a consistent contract for
    the core framework to interact with different problem domains in a fully
    agnostic manner.

    Attributes:
        NAME: Human-readable name of the domain.
        ID: Unique identifier string for the plugin.
        DESCRIPTION: Short description of the domain.
    """

    # Plugin Metadata
    NAME: str = "Base Plugin"
    ID: str = "base"
    DESCRIPTION: str = "Base optimization domain plugin."

    # --------------------------------------------------
    # Required Abstract Methods
    # --------------------------------------------------

    @classmethod
    @abstractmethod
    def get_attribute_schema(cls) -> Dict[str, Dict[str, Any]]:
        """
        Return the attribute schema for the domain.

        The schema defines all possible attributes, their types,
        aggregation rules, evaluation rules, and metadata.

        Returns:
            Dictionary mapping attribute names to their configuration.
        """
        pass

    @classmethod
    @abstractmethod
    def extract_items(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract decision items from the raw JSON data.

        Args:
            data: Raw dictionary loaded from JSON.

        Returns:
            Dictionary mapping item IDs to DecisionItem objects.
        """
        pass

    @classmethod
    @abstractmethod
    def extract_relationships(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relationships and constraints from the raw JSON data.

        Args:
            data: Raw dictionary loaded from JSON.

        Returns:
            Dictionary containing relationships (precedences, couplings,
            exclusions, value dependencies, etc.).
        """
        pass

    @classmethod
    @abstractmethod
    def export_to_dict(cls, model: "OptimizationModel") -> Dict[str, Any]:
        """
        Serialise an OptimizationModel to a domain-specific dictionary.

        Args:
            model: The model to serialise.

        Returns:
            A dictionary in the domain's canonical format.
        """
        pass

    # --------------------------------------------------
    # UI-Agnostic Methods
    # --------------------------------------------------

    @classmethod
    @abstractmethod
    def get_display_name(cls) -> str:
        """
        Return the display name of the domain for the user interface.

        Returns:
            Human-readable name (e.g., 'Next Release Problem (NRP)').
        """
        pass

    @classmethod
    @abstractmethod
    def get_item_label(cls) -> str:
        """
        Return the singular label for a decision item.

        Returns:
            Label string (e.g., 'Requirement' for NRP, 'Product' for Portfolio).
        """
        pass

    @classmethod
    @abstractmethod
    def get_item_label_plural(cls) -> str:
        """
        Return the plural label for decision items.

        Returns:
            Plural label string (e.g., 'Requirements' for NRP).
        """
        pass

    @classmethod
    @abstractmethod
    def get_item_prefix(cls) -> str:
        """
        Return the prefix used for item IDs.

        Returns:
            Prefix string (e.g., 'R' for NRP).
        """
        pass

    @classmethod
    @abstractmethod
    def get_relationship_label(cls, rel_type: str) -> str:
        """
        Return the display label for a specific relationship type.

        Args:
            rel_type: Relationship type key (e.g., 'precedences', 'couplings').

        Returns:
            Display label (e.g., 'Precedence' for NRP, 'Dependency' for Portfolio).
        """
        pass

    @classmethod
    @abstractmethod
    def get_relationship_labels(cls) -> Dict[str, str]:
        """
        Return a dictionary mapping all relationship types to display labels.

        Returns:
            Dictionary of relationship type -> display label.
        """
        pass

    @classmethod
    @abstractmethod
    def get_supported_aggregations(cls) -> List[str]:
        """
        Return the list of supported aggregation rules for multivalued attributes.

        Returns:
            List of rule names (e.g., 'weighted_mean', 'mean', 'sum', 'max', 'min').
        """
        pass

    @classmethod
    @abstractmethod
    def get_supported_couplings(cls) -> List[str]:
        """
        Return the list of supported coupling strategies.

        Returns:
            List of strategy names (e.g., 'sum', 'max', 'min').
        """
        pass

    @classmethod
    @abstractmethod
    def get_generation_config(cls) -> Dict[str, Any]:
        """
        Return the default configuration for synthetic instance generation.

        Returns:
            Dictionary with configuration parameters and their limits.
        """
        pass

    @classmethod
    @abstractmethod
    def generate_synthetic_instance(cls, **kwargs) -> Any:
        """
        Generate a synthetic instance of the domain.

        Args:
            **kwargs: Configuration parameters for generation.

        Returns:
            An OptimizationModel instance.
        """
        pass

    # --------------------------------------------------
    # Optional Methods with Default Implementations
    # --------------------------------------------------

    @classmethod
    def get_read_attributes(cls) -> Dict[str, "AttributeDefinition"]:
        """
        Return the attribute definitions that should be read and aggregated.

        By default, uses the attribute schema to build AttributeDefinition objects.

        Returns:
            Dictionary mapping attribute names to AttributeDefinition objects.
        """
        schema = cls.get_attribute_schema()
        from src.domain.core.definitions import AttributeDefinition

        read_attrs = {}
        for attr_id, info in schema.items():
            read_attrs[attr_id] = AttributeDefinition(
                name=attr_id,
                type=info.get("type", "scalar"),
                description=info.get("description", ""),
                evaluator_group=info.get("evaluator_group"),
                coupling_rule=info.get("coupling_rule", "sum"),
                aggregation_rule=info.get("aggregation_rule", "weighted_mean"),
                evaluation_rule=info.get("evaluation_rule", "sum")
            )
        return read_attrs

    @classmethod
    def get_preprocessing_steps(cls) -> List[tuple]:
        """
        Return the list of preprocessing steps for the pipeline.

        Each step is a tuple (step_name, module_with_run_function).

        Returns:
            List of preprocessing steps (empty by default).
        """
        return []

    @classmethod
    def export_to_json(cls, model: "OptimizationModel", indent: int = 2) -> str:
        """
        Serialise an OptimizationModel to JSON in the domain's format.

        Args:
            model: The model to serialise.
            indent: Indentation level for pretty-printing.

        Returns:
            JSON string.
        """
        data_dict = cls.export_to_dict(model)
        return json.dumps(data_dict, indent=indent, ensure_ascii=False)

    @classmethod
    def clean_attribute_name(cls, attr_name: str) -> str:
        """
        Normalise an attribute name (strip whitespace, convert to lowercase).

        Args:
            attr_name: Raw attribute name.

        Returns:
            Normalised attribute name.
        """
        if not attr_name:
            return ""
        return attr_name.strip().lower()

    @classmethod
    def identify_decision_variables(cls, df: pd.DataFrame) -> List[str]:
        """
        Identify decision variable columns in a results DataFrame.

        Args:
            df: DataFrame containing results.

        Returns:
            List of column names that represent decision variables.
        """
        return []

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

        Args:
            df: DataFrame containing results.
            model: Optional model instance for context.
            is_calculated: Whether the front was calculated (vs. loaded).
            configured_objectives: Objectives configured in the problem.
            decision_vars: Pre-identified decision variable columns.

        Returns:
            List of column names that represent objectives.
        """
        return []

    # --------------------------------------------------
    # Enrichment Methods (New)
    # --------------------------------------------------

    @classmethod
    @abstractmethod
    def get_enrichment_indicators(cls) -> Dict[str, Dict[str, Any]]:
        """
        Return available enrichment indicators for the domain.

        Each indicator must have:
            - "description": str
            - "required_attributes": List[str] (attributes needed in the DataFrame)

        Returns:
            Dictionary mapping indicator names to their metadata.
        """
        pass

    @classmethod
    @abstractmethod
    def compute_enrichment(
        cls,
        df: pd.DataFrame,
        indicators: List[str],
        decision_var_prefix: str = "req_"
    ) -> pd.DataFrame:
        """
        Compute the requested enrichment indicators on the DataFrame.

        Args:
            df: Input DataFrame containing solutions.
            indicators: List of indicator names to compute.
            decision_var_prefix: Prefix used for decision variable columns.

        Returns:
            DataFrame with the computed indicators added as columns.
        """
        pass