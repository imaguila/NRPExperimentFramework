# Location: src/analysis/lenses/preference.py
"""
Preference Lens Module (Core).

Multi-Criteria Decision Making (MCDM) algorithms:
- Weighted Sum
- TOPSIS
- VIKOR
- Reference Point
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.analysis.lenses.base_analysis import BaseLens
from src.analysis.lenses.utils_metrics import minmax_normalize_dataframe


class PreferenceLens(BaseLens):
    """Lente de Preferencia (MCDM)."""
    name: str = "Preference"
    category: str = "Preference"
    description: str = "Ranks and filters solutions using multi-criteria decision making (MCDM)."

    def get_schema(
        self,
        df: pd.DataFrame,
        dimensions: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Devuelve el esquema de parámetros que la UI debe renderizar.
        """
        if df is None or df.empty:
            return []

        filtered_dimensions = [
            d for d in dimensions
            if not d.startswith(("req_", "x_", "var_", "item_"))
            and d.lower() not in ("id", "selected_ids", "selected_ids_str", "is_feasible")
        ]

        return [
            {
                "key": "method",
                "label": "Scoring Method",
                "type": "select",
                "options": ["Weighted Sum", "TOPSIS", "VIKOR", "Reference Point"],
                "default": "Weighted Sum",
            },
            {
                "key": "maximize",
                "label": "Metrics to Maximize",
                "type": "multiselect",
                "options": filtered_dimensions,
                "default": filtered_dimensions[:1] if filtered_dimensions else [],
            },
            {
                "key": "minimize",
                "label": "Metrics to Minimize",
                "type": "multiselect",
                "options": filtered_dimensions,
                "default": filtered_dimensions[1:2] if len(filtered_dimensions) > 1 else [],
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
        Aplica el método MCDM seleccionado y devuelve las Top N soluciones.
        """
        if df is None or df.empty:
            return df

        result = df.copy()
        maximize = params.get("maximize", [])
        minimize = params.get("minimize", [])

        valid_max = [m for m in maximize if m in result.columns]
        valid_min = [m for m in minimize if m in result.columns and m not in valid_max]
        criteria = valid_max + valid_min

        if not criteria:
            return result

        method = params.get("method", "Weighted Sum")
        top_n = min(params.get("top_n", len(result)), len(result))

        if method == "Weighted Sum":
            score = self._weighted_sum(result, valid_max, valid_min)
        elif method == "TOPSIS":
            score = self._topsis(result, valid_max, valid_min)
        elif method == "VIKOR":
            score = self._vikor(result, valid_max, valid_min)
        elif method == "Reference Point":
            score = self._reference_point(result, valid_max, valid_min)
        else:
            return result

        result["preference_score"] = score
        result = result.sort_values("preference_score", ascending=False).copy()
        result["preference_rank"] = range(1, len(result) + 1)
        result["preference_method"] = method

        return result.head(top_n)

    # ------------------------------------------------------------------
    # Algoritmos MCDM Internos (Vectorizados)
    # ------------------------------------------------------------------

    def _weighted_sum(self, df: pd.DataFrame, maximize: List[str], minimize: List[str]) -> pd.Series:
        criteria = maximize + minimize
        norm_df = minmax_normalize_dataframe(df, criteria)
        score = pd.Series(0.0, index=df.index)
        weight = 1.0 / len(criteria)
        for metric in criteria:
            val = norm_df[metric] if metric in maximize else (1.0 - norm_df[metric])
            score += weight * val
        return score

    def _topsis(self, df: pd.DataFrame, maximize: List[str], minimize: List[str]) -> pd.Series:
        criteria = maximize + minimize
        vals = df[criteria].to_numpy(dtype=float)

        norms = np.linalg.norm(vals, axis=0)
        norms[norms == 0] = 1.0
        norm_vals = (vals / norms) * (1.0 / len(criteria))

        ideal = np.zeros(len(criteria))
        anti_ideal = np.zeros(len(criteria))

        for idx, metric in enumerate(criteria):
            if metric in maximize:
                ideal[idx] = norm_vals[:, idx].max()
                anti_ideal[idx] = norm_vals[:, idx].min()
            else:
                ideal[idx] = norm_vals[:, idx].min()
                anti_ideal[idx] = norm_vals[:, idx].max()

        d_plus = np.linalg.norm(norm_vals - ideal, axis=1)
        d_minus = np.linalg.norm(norm_vals - anti_ideal, axis=1)

        denom = d_plus + d_minus
        scores = np.where(denom != 0, d_minus / denom, 0.0)
        return pd.Series(scores, index=df.index)

    def _vikor(self, df: pd.DataFrame, maximize: List[str], minimize: List[str], v: float = 0.5) -> pd.Series:
        criteria = maximize + minimize
        weight = 1.0 / len(criteria)
        regret = pd.DataFrame(index=df.index)

        for metric in criteria:
            best, worst = (df[metric].max(), df[metric].min()) if metric in maximize else (df[metric].min(), df[metric].max())
            denom = abs(best - worst)
            regret[metric] = 0.0 if denom == 0 else weight * abs(best - df[metric]) / denom

        s_val = regret.sum(axis=1)
        r_val = regret.max(axis=1)

        s_range = s_val.max() - s_val.min()
        s_norm = (s_val - s_val.min()) / s_range if s_range > 0 else 0.0

        r_range = r_val.max() - r_val.min()
        r_norm = (r_val - r_val.min()) / r_range if r_range > 0 else 0.0

        q_val = v * s_norm + (1.0 - v) * r_norm
        return 1.0 - q_val

    def _reference_point(self, df: pd.DataFrame, maximize: List[str], minimize: List[str]) -> pd.Series:
        criteria = maximize + minimize
        norm_df = minmax_normalize_dataframe(df, criteria)
        oriented = pd.DataFrame(index=df.index)

        for metric in criteria:
            oriented[metric] = norm_df[metric] if metric in maximize else (1.0 - norm_df[metric])

        distances = np.linalg.norm(1.0 - oriented.to_numpy(dtype=float), axis=1)
        max_dist = distances.max()
        scores = 1.0 - (distances / max_dist) if max_dist > 0 else np.ones(len(df))
        return pd.Series(scores, index=df.index)


# ===================================================================
# EXPOSICIÓN A NIVEL DE MÓDULO (Para compatibilidad con el registro)
# ===================================================================

_lens_instance = PreferenceLens()


def apply(
    df: pd.DataFrame,
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Punto de entrada invocado directamente por el pipeline."""
    return _lens_instance.apply(df, params, context)