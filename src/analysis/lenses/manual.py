# Location: src/analysis/lenses/manual.py
"""
Manual Selection Lens.

Pure analytical filter for isolating candidate solutions by identifier.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from src.analysis.lenses.base_analysis import BaseLens


class ManualLens(BaseLens):
    """Lente de Selección Manual: Aísla soluciones mediante IDs explícitos."""
    name: str = "Manual Selection"
    category: str = "Manual"
    description: str = "Isolates specific candidate solutions using explicit ID selection."

    def apply(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Filtra el DataFrame seleccionando únicamente las soluciones con los IDs especificados.
        """
        if df is None or df.empty:
            return df

        selected_ids: List[str] = params.get("selected_ids", [])
        if not selected_ids or "id" not in df.columns:
            return df.iloc[0:0].copy()

        return df[df["id"].isin(selected_ids)].copy()