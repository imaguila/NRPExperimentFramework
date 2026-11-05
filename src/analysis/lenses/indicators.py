# Location: src/analysis/lenses/indicators.py
"""
Indicator Lens Module (Core).

Provides multi-criteria selection methods based on domain indicators:
1. Top-N Matches: Aggregates top solutions across individual target dimensions
   and allows filtering/selecting specific match count groups.
2. Non-Dominated Sorting: Identifies Pareto-optimal solutions within the enriched 
   indicator space.

Zero Streamlit dependencies.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.analysis.lenses.base_analysis import BaseLens


class IndicatorLens(BaseLens):
    """Lente de Indicadores: Top-N Matches y Filtrado Pareto."""
    name: str = "Indicator"
    category: str = "Multi-Criteria"
    description: str = "Selects Pareto non-dominated solutions or isolates candidates meeting Top-N criteria targets."

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

        # Filtrar dimensiones
        filtered_dimensions = [
            d for d in dimensions
            if not d.startswith(("req_", "x_", "var_", "item_"))
            and d.lower() not in ("id", "selected_ids", "selected_ids_str", "is_feasible")
        ]

        if not filtered_dimensions:
            return []

        default_max = filtered_dimensions[: min(2, len(filtered_dimensions))]
        default_min = [filtered_dimensions[2]] if len(filtered_dimensions) > 2 else []

        # --- CORRECCIÓN: Asegurar que context no sea None ---
        safe_context = context if context is not None else {}
        method = safe_context.get("method", "Top-N Matches")
        # ----------------------------------------------------

        schema: List[Dict[str, Any]] = [
            {
                "key": "method",
                "label": "Indicator Method",
                "type": "select",
                "options": ["Top-N Matches", "Non-dominated"],
                "default": "Top-N Matches",
            },
            {
                "key": "maximize",
                "label": "Maximize Criteria",
                "type": "multiselect",
                "options": filtered_dimensions,
                "default": default_max,
            },
            {
                "key": "minimize",
                "label": "Minimize Criteria",
                "type": "multiselect",
                "options": filtered_dimensions,
                "default": default_min,
            },
        ]

        # Si es Top-N Matches, añadir controles adicionales
        if method == "Top-N Matches":
            max_top_n = max(2, min(50, len(df))) if df is not None and not df.empty else 10
            schema.append({
                "key": "top_n",
                "label": "Top-N Cutoff per Criteria",
                "type": "slider",
                "min": 1,
                "max": max_top_n,
                "default": min(5, max_top_n),
            })

            schema.append({
                "key": "match_filter",
                "label": "Match Strategy",
                "type": "select",
                "options": ["All", "Highest", "At least", "Exact"],
                "default": "All",
            })

            max_criteria = len(default_max) + len(default_min)
            max_criteria = max(1, max_criteria)

            # --- CORRECCIÓN: Usar safe_context en lugar de context ---
            if safe_context.get("match_filter", "All") in ["At least", "Exact"]:
                schema.append({
                    "key": "target_matches",
                    "label": "Target Match Count",
                    "type": "slider",
                    "min": 1,
                    "max": max_criteria,
                    "default": 1,
                })
            # ----------------------------------------------------

            # Selector de grupos
            group_options = ["All Groups"] + [f"Matches = {i}" for i in range(max_criteria, 0, -1)]
            schema.append({
                "key": "target_group",
                "label": "Select SOI Match Group",
                "type": "select",
                "options": group_options,
                "default": "All Groups",
            })

        return schema

    def apply(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Aplica el método seleccionado (Top-N Matches o Non-dominated).
        """
        if df is None or df.empty:
            return df

        result = df.copy()
        maximize, minimize, criteria = self._sanitize_criteria(
            result,
            params.get("maximize", []),
            params.get("minimize", []),
        )

        if not criteria:
            return result

        method = params.get("method", "Top-N Matches")

        if method == "Top-N Matches":
            top_n = int(params.get("top_n", min(5, len(result))))
            match_filter = params.get("match_filter", "All")
            target_matches = int(params.get("target_matches", 1))
            target_group = params.get("target_group", "All Groups")

            res_df = self._apply_top_n_matches(
                result, maximize, minimize, top_n, match_filter, target_matches
            )

            # Filtrado secundario por grupo explícito (SOI)
            if not res_df.empty and target_group != "All Groups" and "Matches = " in target_group:
                try:
                    c_val = int(target_group.replace("Matches = ", "").strip())
                    res_df = res_df[res_df["domain_match_count"] == c_val].copy()
                except ValueError:
                    pass

            return res_df

        if method == "Non-dominated":
            return self._apply_non_dominated(result, maximize, minimize)

        return result

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_criteria(
        df: pd.DataFrame, maximize: List[str], minimize: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        valid_max = [m for m in maximize if m in df.columns]
        valid_min = [m for m in minimize if m in df.columns and m not in valid_max]
        criteria = valid_max + valid_min
        return valid_max, valid_min, criteria

    @staticmethod
    def _build_group_labels_from_count(
        result: pd.DataFrame, count_column: str
    ) -> pd.DataFrame:
        result["group_base"] = result[count_column].apply(
            lambda count: f"Matches = {count}"
        )
        group_sizes = result["group_base"].value_counts().to_dict()
        result["group_label"] = result["group_base"].apply(
            lambda grp: f"{grp} (n={group_sizes.get(grp, 0)})"
        )
        return result

    def _apply_top_n_matches(
        self,
        df: pd.DataFrame,
        maximize: List[str],
        minimize: List[str],
        top_n: int,
        match_filter: str = "All",
        target_matches: int = 1,
    ) -> pd.DataFrame:
        result = df.copy()
        criteria = maximize + minimize

        if not criteria:
            return result

        id_col = "id" if "id" in result.columns else ("ID" if "ID" in result.columns else None)
        if not id_col:
            result["id"] = range(len(result))
            id_col = "id"

        effective_top_n = min(max(1, top_n), len(result))
        ranked_subsets: List[pd.DataFrame] = []

        for metric in maximize:
            sub = (
                result.sort_values(metric, ascending=False)
                .head(effective_top_n)[[id_col]]
                .assign(matched_metric=metric, goal="Maximize")
            )
            ranked_subsets.append(sub)

        for metric in minimize:
            sub = (
                result.sort_values(metric, ascending=True)
                .head(effective_top_n)[[id_col]]
                .assign(matched_metric=metric, goal="Minimize")
            )
            ranked_subsets.append(sub)

        if not ranked_subsets:
            return result

        matches = pd.concat(ranked_subsets, ignore_index=True)

        counts = (
            matches.groupby(id_col)
            .size()
            .reset_index(name="domain_match_count")
        )

        matched_metrics = (
            matches.groupby(id_col)["matched_metric"]
            .apply(lambda vals: ", ".join(sorted(set(vals))))
            .reset_index(name="domain_matched_metrics")
        )

        result = result.merge(counts, on=id_col, how="left").merge(
            matched_metrics, on=id_col, how="left"
        )

        result["domain_match_count"] = result["domain_match_count"].fillna(0).astype(int)
        result["domain_matched_metrics"] = result["domain_matched_metrics"].fillna("")

        # Filtro base: al menos 1 coincidencia
        result = result[result["domain_match_count"] > 0].copy()

        if result.empty:
            return result

        # Filtrado por estrategia
        if match_filter == "Highest":
            max_c = result["domain_match_count"].max()
            result = result[result["domain_match_count"] == max_c].copy()
        elif match_filter == "At least":
            result = result[result["domain_match_count"] >= target_matches].copy()
        elif match_filter == "Exact":
            result = result[result["domain_match_count"] == target_matches].copy()

        if result.empty:
            return result

        result = self._build_group_labels_from_count(result, "domain_match_count")
        result = result.sort_values(
            ["domain_match_count", id_col], ascending=[False, True]
        ).copy()

        result["domain_rank"] = range(1, len(result) + 1)
        result["indicator_method"] = "Top-N Matches"
        result["indicator_top_n"] = effective_top_n

        return result

    def _apply_non_dominated(
        self, df: pd.DataFrame, maximize: List[str], minimize: List[str]
    ) -> pd.DataFrame:
        result = df.copy()
        criteria = maximize + minimize

        if not criteria:
            return result

        work = result[criteria].copy()

        # Invertir métricas a minimizar para maximización estricta
        for metric in minimize:
            work[metric] = -work[metric]

        values = work.to_numpy(dtype=float)
        n_samples = len(values)
        is_nondominated = np.ones(n_samples, dtype=bool)

        for i in range(n_samples):
            current = values[i]
            for j in range(n_samples):
                if i == j:
                    continue
                challenger = values[j]
                if np.all(challenger >= current) and np.any(challenger > current):
                    is_nondominated[i] = False
                    break

        result["indicator_nondominated"] = is_nondominated
        result = result[result["indicator_nondominated"]].copy()

        if result.empty:
            return result

        id_col = "id" if "id" in result.columns else ("ID" if "ID" in result.columns else None)

        result["indicator_method"] = "Non-dominated"
        result["domain_match_count"] = len(criteria)
        result["domain_matched_metrics"] = ", ".join(criteria)
        result["group_base"] = "Non-dominated"
        result["group_label"] = f"Non-dominated (n={len(result)})"

        if id_col:
            result = result.sort_values(id_col, ascending=True).copy()

        result["domain_rank"] = range(1, len(result) + 1)

        return result


# ===================================================================
# EXPOSICIÓN A NIVEL DE MÓDULO (Para compatibilidad con el registro)
# ===================================================================

_lens_instance = IndicatorLens()


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