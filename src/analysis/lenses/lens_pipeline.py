# Location: src/analysis/lenses/lens_pipeline.py
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.analysis.lenses.registry import get_lens
from src.domain.core.paretofront import ParetoFront
from src.domain.core.collection import SolutionSet


def evaluate_lens(
    front: ParetoFront,
    lens_name: str,
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> SolutionSet:
    """
    Ejecuta de forma polimórfica la lente seleccionada sobre un ParetoFront.
    
    Args:
        front: El frente de Pareto a analizar.
        lens_name: Nombre de la lente a aplicar.
        params: Parámetros de configuración de la lente.
        context: Contexto adicional para metadatos.
    
    Returns:
        SolutionSet: El conjunto de soluciones resultante (SOI).
    """
    # 1. Caso especial: "None" -> devolver el frente completo como SOI
    if lens_name in ["None", "None / Full Set", "", None]:
        return front.to_solution_set(name=front.name or "Full Set", category="soi")

    # 2. Obtener la lente del registro
    lens = get_lens(lens_name)
    if lens is None:
        return front.to_solution_set(name=front.name or "Full Set", category="soi")

    # 3. Convertir el ParetoFront a DataFrame
    df = front.to_dataframe()
    if df.empty:
        return front.to_solution_set(name="Empty", category="soi")

    # 4. Aplicar la lente (trabaja con DataFrame)
    processed_df = lens.apply(df, params, context=context)

    # 5. Convertir el DataFrame resultante a SolutionSet
    #    Para ello, necesitamos extraer los IDs de las soluciones del DataFrame
    if processed_df.empty:
        # Si no quedan soluciones, devolvemos un conjunto vacío
        return front.to_solution_set(name="Empty SOI", category="soi")

    # Extraer los IDs de las soluciones del DataFrame
    id_col = "id" if "id" in processed_df.columns else ("ID" if "ID" in processed_df.columns else None)
    if id_col:
        selected_ids = set(processed_df[id_col].astype(str).tolist())
    else:
        # Si no hay columna ID, usar el índice del DataFrame
        selected_ids = set(processed_df.index.astype(str).tolist())

    # Construir un nuevo ParetoFront solo con las soluciones seleccionadas
    filtered_solutions = [sol for sol in front.solutions if sol.id in selected_ids]
    filtered_front = ParetoFront(filtered_solutions)

    # Crear un SolutionSet con metadatos
    return filtered_front.to_solution_set(
        name=f"{lens_name} SOI",
        category="soi",
        metadata={
            "lens_name": lens_name,
            "params": params,
            "context": context or {},
        }
    )


def get_lens_schema(
    lens_name: str,
    df: pd.DataFrame,
    dimensions: List[str],
    params: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Obtiene el esquema de parámetros UI requeridos por la lente.
    
    Args:
        lens_name: Nombre de la lente.
        df: DataFrame actual (para inferir opciones).
        dimensions: Dimensiones disponibles.
        params: Parámetros actuales (para valores por defecto).
        context: Contexto adicional.
    
    Returns:
        List[Dict[str, Any]]: Esquema de parámetros para la UI.
    """
    if lens_name in ["None", "None / Full Set", "", None]:
        return []

    lens = get_lens(lens_name)
    if lens is None:
        return []

    return lens.__class__.get_schema(df, dimensions, params=params, context=context)