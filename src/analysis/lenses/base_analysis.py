from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd


class BaseLens(ABC):
    """
    Clase base abstracta para todas las Lentes Analíticas.
    
    Recibe un DataFrame (o contexto con múltiples frentes/SOIs), aplica una 
    estrategia de filtrado/agrupamiento y retorna el DataFrame resultante.
    """
    name: str
    category: str
    description: str

    @classmethod
    def get_schema(
        cls,
        df: pd.DataFrame,
        dimensions: List[str],
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Devuelve la especificación UI (esquema de parámetros) de la lente."""
        return []

    @abstractmethod
    def apply(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Aplica la lógica analítica y retorna el DataFrame filtrado o etiquetado."""
        pass