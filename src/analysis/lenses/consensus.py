# Location: src/analysis/lenses/consensus.py
"""
Consensus Lens Module (Headless Core).

Aggregates multiple saved Sets of Interest (SOIs) into a unified consensus
model using threshold-based voting. The lens expects a context containing
a list of SOI objects (with attribute `solution_ids`) and produces a DataFrame
with consensus scores, support counts, and group labels.

Pure Python & Pandas module — Framework agnostic.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from src.analysis.lenses.base_analysis import BaseLens


class ConsensusLens(BaseLens):
    """Lente de Consenso: Combina múltiples SOIs mediante votación."""
    name: str = "Consensus"
    category: str = "Combination"
    description: str = (
        "Combines multiple saved SOIs into a consensus set using voting."
    )

    def get_schema(
        self,
        df: pd.DataFrame,
        dimensions: List[str],
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Devuelve el esquema de parámetros que la UI debe renderizar.
        El contexto debe contener 'saved_sois' con la lista de objetos SOI.
        """
        if df is None or df.empty:
            return []

        # Obtener nombres de SOIs desde el contexto
        saved_sois = (context or {}).get("saved_sois", [])
        if not saved_sois:
            return []  # No hay SOIs para elegir

        soi_names = [soi.name for soi in saved_sois if hasattr(soi, "name")]

        # Esquema base: selección de SOIs y umbral
        schema = [
            {
                "key": "selected_sois",
                "label": "Select SOIs to Include",
                "type": "multiselect",
                "options": soi_names,
                "default": soi_names[: min(3, len(soi_names))],
            },
            {
                "key": "threshold",
                "label": "Consensus Threshold",
                "type": "slider_float",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "default": 0.5,
            },
        ]

        # Opcional: permitir filtrar por grupo de soporte (si ya hay un resultado)
        # Esta parte se podría añadir después de aplicar la lente, pero
        # lo dejamos para la UI a través del selector de grupos general.

        return schema

    def apply(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Aplica el cálculo de consenso al DataFrame activo.
        Retorna un DataFrame con las soluciones que superan el umbral,
        enriquecido con columnas de consenso.
        """
        if df is None or df.empty:
            return df

        # 1. Obtener las SOIs seleccionadas desde el contexto
        saved_sois = (context or {}).get("saved_sois", [])
        selected_names = params.get("selected_sois", [])
        if not selected_names:
            return df

        # Filtrar solo las SOIs cuyo nombre coincide
        selected_sois = [
            soi for soi in saved_sois
            if hasattr(soi, "name") and soi.name in selected_names
        ]
        if len(selected_sois) < 2:
            # Si hay menos de 2 SOIs, no hay consenso significativo
            return df.iloc[0:0].copy()

        # 2. Extraer IDs de cada SOI
        all_solution_ids = []
        for soi in selected_sois:
            # Se asume que SOI tiene un atributo `solution_ids` (lista de IDs)
            ids = getattr(soi, "solution_ids", [])
            all_solution_ids.append(set(ids))

        # 3. Calcular soporte por solución
        support: Dict[str, int] = {}
        supporting_sois: Dict[str, List[str]] = {}

        for soi_idx, ids in enumerate(all_solution_ids):
            soi_name = selected_sois[soi_idx].name
            for sol_id in ids:
                sol_id_str = str(sol_id).strip()
                support[sol_id_str] = support.get(sol_id_str, 0) + 1
                supporting_sois.setdefault(sol_id_str, []).append(soi_name)

        # 4. Construir DataFrame de soporte
        n_sois = len(selected_sois)
        rows = []
        for sol_id, count in support.items():
            rows.append({
                "id": sol_id,
                "consensus_support_count": count,
                "consensus_score": count / n_sois,
                "consensus_supporting_sois": ", ".join(
                    sorted(supporting_sois.get(sol_id, []))
                ),
            })
        support_df = pd.DataFrame(rows)

        if support_df.empty:
            return df.iloc[0:0].copy()

        # 5. Filtrar por umbral
        threshold = params.get("threshold", 0.5)
        support_df = support_df[support_df["consensus_score"] >= (threshold - 1e-9)]

        if support_df.empty:
            return df.iloc[0:0].copy()

        # 6. Unir con el DataFrame original (por columna 'id')
        # Normalizar tipos de ID para evitar fallos
        df_clean = df.copy()
        if "id" not in df_clean.columns:
            # Si no hay columna 'id', usar el índice como ID
            df_clean["id"] = df_clean.index.astype(str)

        # Asegurar que ambos lados tengan IDs como string
        df_clean["id"] = df_clean["id"].astype(str).str.strip()
        support_df["id"] = support_df["id"].astype(str).str.strip()

        result = df_clean.merge(support_df, on="id", how="inner")

        if result.empty:
            return df.iloc[0:0].copy()

        # 7. Añadir etiquetas de grupo
        result["group_base"] = result["consensus_support_count"].apply(
            lambda count: f"Support = {count}/{n_sois}"
        )
        group_sizes = result["group_base"].value_counts().to_dict()
        result["group_label"] = result["group_base"].apply(
            lambda grp: f"{grp} (n={group_sizes.get(grp, 0)})"
        )

        # 8. Metadatos adicionales
        result["consensus_method"] = "Consensus Threshold"
        result["consensus_threshold"] = threshold
        result["consensus_source_sois"] = ", ".join(selected_names)

        # Ordenar por soporte descendente
        result = result.sort_values(
            ["consensus_score", "consensus_support_count"],
            ascending=[False, False]
        )
        result["consensus_rank"] = range(1, len(result) + 1)

        return result


# ===================================================================
# EXPOSICIÓN A NIVEL DE MÓDULO (Para compatibilidad con el registro)
# ===================================================================

_lens_instance = ConsensusLens()


def apply(
    df: pd.DataFrame,
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Punto de entrada invocado directamente por el pipeline."""
    return _lens_instance.apply(df, params, context)


def get_schema(
    df: pd.DataFrame,
    dimensions: List[str],
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Punto de entrada para obtener el esquema de la lente."""
    return _lens_instance.get_schema(df, dimensions, params=params)