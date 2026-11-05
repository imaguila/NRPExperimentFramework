"""
Generic synthetic instance generator.

This module must remain domain-agnostic.
Domain-specific generation belongs in the plugin layer, e.g. NRP.
"""

from typing import Any, Dict, Optional, Type

from src.domain.core.model import OptimizationModel
from src.domain.plugins.base_domain import DomainPlugin


class SyntheticInstanceGenerator:
    """
    Generic factory placeholder.
    Do not put NRP-specific logic here.
    """

    @staticmethod
    def generate_instance(
        num_items: int,
        attribute_configs: Dict[str, Dict[str, Any]],
        model_name: str = "Synthetic_Instance",
        id_prefix: str = "R",
        value_dependencies: Optional[list] = None,
        plugin: Optional[Type[DomainPlugin]] = None,
    ) -> OptimizationModel:
        raise NotImplementedError(
            "Concrete synthetic generation belongs to a domain plugin "
            "(for example: src/domain/plugins/nrp_generation.py)."
        )