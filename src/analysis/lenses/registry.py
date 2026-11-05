# Location: src/analysis/lenses/registry.py
"""
Analytical Lenses Registry.

Maps lens names to their respective processing modules.
"""

from typing import Dict, List, Optional, Type

from src.analysis.lenses.base_analysis import BaseLens
from src.analysis.lenses.manual import ManualLens
from src.analysis.lenses.preference import PreferenceLens
from src.analysis.lenses.efficiency import EfficiencyLens
from src.analysis.lenses.indicators import IndicatorLens
from src.analysis.lenses.diversity import DiversityLens
from src.analysis.lenses.consensus import ConsensusLens  


# Registro centralizado de lentes (se puede extender dinámicamente)
_LENS_REGISTRY: Dict[str, Type[BaseLens]] = {
    ManualLens.name: ManualLens,
    PreferenceLens.name: PreferenceLens,
    EfficiencyLens.name: EfficiencyLens,
    IndicatorLens.name: IndicatorLens,
    DiversityLens.name: DiversityLens,
    ConsensusLens.name: ConsensusLens,  # <--- AÑADIDO
}

def get_lens_names() -> List[str]:
    """Retorna la lista de nombres de Lentes disponibles."""
    return ["None"] + list(_LENS_REGISTRY.keys())


def get_lens(name: str) -> Optional[BaseLens]:
    """Obtiene la instancia de la Lente seleccionada por su nombre."""
    lens_cls = _LENS_REGISTRY.get(name)
    if lens_cls is None:
        return None
    return lens_cls()


def register_lens(lens_cls: Type[BaseLens]) -> None:
    """Registra una nueva lente dinámicamente."""
    if issubclass(lens_cls, BaseLens) and hasattr(lens_cls, "name"):
        _LENS_REGISTRY[lens_cls.name] = lens_cls
    else:
        raise TypeError("La lente registrada debe ser una subclase de BaseLens.")