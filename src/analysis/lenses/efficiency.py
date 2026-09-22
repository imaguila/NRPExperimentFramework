# Location: src/analysis/lenses/efficiency.py
"""
Efficiency Lens Module (Core).

Ranks candidate solutions based on benefit-cost trade-offs using raw ratios,
min-max normalized efficiency, composite cost aggregation, or Euclidean distance
to ideal target states in objective space.

Zero Streamlit dependencies.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.analysis.lenses.base_analysis import BaseLens
from src.analysis.lenses.utils_metrics import minmax_normalize

EPS: float = 1e-9


class EfficiencyLens(BaseLens):
    """Lente de Eficiencia: rankea soluciones por ratio beneficio/coste."""
    name: str = "Efficiency"
    category: str = "Trade-off"
    description: str = "Ranks solutions by benefit-to-cost efficiency and trade-off metrics."

    def get_schema(
        self,
        df: pd.DataFrame,
        dimensions: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Devuelve el esquema de parámetros que la UI debe renderizar.
        """
        if df is None or df.empty or not dimensions:
            return []

        # Filtrar dimensiones (excluir variables de decisión y metadatos)
        filtered_dimensions = [
            d for d in dimensions
            if not d.startswith(("req_", "x_", "var_", "item_"))
            and d.lower() not in ("id", "selected_ids", "selected_ids_str", "is_feasible")
        ]

        if len(filtered_dimensions) < 2:
            return []

        # Valores por defecto
        default_benefit = filtered_dimensions[0]
        cost_options = [d for d in filtered_dimensions if d != default_benefit]

        return [
            {
                "key": "method",
                "label": "Efficiency Engine",
                "type": "select",
                "options": [
                    "Benefit/Cost Ratio",
                    "Normalized Ratio",
                    "Distance to Ideal",
                    "Composite Cost Ratio",
                ],
                "default": "Benefit/Cost Ratio",
            },
            {
                "key": "benefit",
                "label": "Benefit Metric (Maximize)",
                "type": "select",
                "options": filtered_dimensions,
                "default": default_benefit,
            },
            {
                "key": "cost",
                "label": "Cost Metric (Minimize)",
                "type": "select" if len(cost_options) > 0 else "hidden",
                "options": cost_options,
                "default": cost_options[0] if cost_options else None,
            },
            {
                "key": "top_n",
                "label": "Top N Solutions",
                "type": "slider",
                "min": 1,
                "max": len(df),
                "default": min(15, len(df)),
            },
        ]

    def apply(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Aplica el motor de eficiencia seleccionado y devuelve las Top N soluciones.
        """
        if df is None or df.empty:
            return df

        result = df.copy()
        method = params.get("method", "Benefit/Cost Ratio")
        benefit = params.get("benefit")
        cost = params.get("cost")
        top_n = min(params.get("top_n", len(result)), len(result))

        if benefit is None or benefit not in result.columns:
            return result

        cost_metrics = self._resolve_cost_metrics(result, benefit, cost)
        if not cost_metrics:
            return result

        if method == "Benefit/Cost Ratio":
            score = self._benefit_cost_ratio(result, benefit, cost_metrics)
        elif method == "Normalized Ratio":
            score = self._normalized_ratio(result, benefit, cost_metrics)
        elif method == "Distance to Ideal":
            score = self._distance_to_ideal(result, benefit, cost_metrics)
        elif method == "Composite Cost Ratio":
            score = self._composite_cost_ratio(result, benefit, cost_metrics)
            result["efficiency_costs"] = ", ".join(cost_metrics)
        else:
            return result

        result["efficiency_score"] = score
        result = result.sort_values("efficiency_score", ascending=False).copy()
        result["efficiency_rank"] = range(1, len(result) + 1)
        result["efficiency_method"] = method
        result["efficiency_benefit"] = benefit
        result["efficiency_primary_cost"] = cost_metrics[0]

        return result.head(top_n)

    # ------------------------------------------------------------------
    # Motores de cálculo internos
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_cost_metrics(
        result: pd.DataFrame,
        benefit: str,
        cost: Optional[Union[str, List[str]]],
    ) -> List[str]:
        if cost is None:
            return []
        if isinstance(cost, str):
            cost_metrics = [cost]
        else:
            cost_metrics = [c for c in cost if c in result.columns]
        return [c for c in cost_metrics if c != benefit]

    def _benefit_cost_ratio(self, result: pd.DataFrame, benefit: str, cost_metrics: List[str]) -> pd.Series:
        cost_metric = cost_metrics[0]
        safe_cost = result[cost_metric].replace(0, EPS)
        return result[benefit] / safe_cost

    def _normalized_ratio(self, result: pd.DataFrame, benefit: str, cost_metrics: List[str]) -> pd.Series:
        cost_metric = cost_metrics[0]
        benefit_norm = minmax_normalize(result[benefit])
        cost_norm = minmax_normalize(result[cost_metric])
        return benefit_norm / (cost_norm + EPS)

    def _distance_to_ideal(self, result: pd.DataFrame, benefit: str, cost_metrics: List[str]) -> pd.Series:
        cost_metric = cost_metrics[0]
        benefit_norm = minmax_normalize(result[benefit])
        cost_norm = minmax_normalize(result[cost_metric])
        distance_to_ideal = ((1.0 - benefit_norm) ** 2 + (cost_norm) ** 2) ** 0.5
        max_distance = 2.0 ** 0.5
        return 1.0 - (distance_to_ideal / max_distance)

    def _composite_cost_ratio(self, result: pd.DataFrame, benefit: str, cost_metrics: List[str]) -> pd.Series:
        benefit_norm = minmax_normalize(result[benefit])
        composite_cost = pd.Series(0.0, index=result.index)
        for cost_metric in cost_metrics:
            composite_cost += minmax_normalize(result[cost_metric])
        composite_cost /= len(cost_metrics)
        return benefit_norm / (composite_cost + EPS)


# ===================================================================
# EXPOSICIÓN A NIVEL DE MÓDULO (Para compatibilidad con el registro)
# ===================================================================

_lens_instance = EfficiencyLens()


def apply(
    df: pd.DataFrame,
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Punto de entrada invocado directamente por el pipeline."""
    return _lens_instance.apply(df, params, context)


def get_schema(
    df: pd.DataFrame, dimensions: List[str], params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Punto de entrada para obtener el esquema de la lente."""
    return _lens_instance.get_schema(df, dimensions, params=params)