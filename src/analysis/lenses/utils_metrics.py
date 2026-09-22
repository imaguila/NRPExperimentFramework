"""
Utility functions for metric normalization and common calculations.
Used exclusively by analytical lenses.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def minmax_normalize(series: pd.Series) -> pd.Series:
    """
    Normaliza una serie al rango [0, 1] usando min‑max.
    Si todos los valores son iguales, devuelve una serie de ceros.
    """
    min_v = series.min()
    max_v = series.max()
    if max_v > min_v:
        return (series - min_v) / (max_v - min_v)
    return pd.Series(0.0, index=series.index)


def minmax_normalize_dataframe(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normaliza columnas específicas de un DataFrame al rango [0, 1].
    Retorna un nuevo DataFrame con las columnas normalizadas.
    """
    norm_df = df.copy()
    for col in columns:
        norm_df[col] = minmax_normalize(df[col])
    return norm_df


def zscore_normalize(series: pd.Series) -> pd.Series:
    """
    Normaliza una serie usando z‑score (media 0, desviación 1).
    Si la desviación es cero, devuelve una serie de ceros.
    """
    mean = series.mean()
    std = series.std()
    if std > 0:
        return (series - mean) / std
    return pd.Series(0.0, index=series.index)


def zscore_normalize_dataframe(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normaliza columnas específicas de un DataFrame usando z‑score.
    Retorna un nuevo DataFrame con las columnas normalizadas.
    """
    norm_df = df.copy()
    for col in columns:
        norm_df[col] = zscore_normalize(df[col])
    return norm_df


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calcula la distancia euclídea entre dos vectores."""
    return np.linalg.norm(vec1 - vec2)


def normalize_to_range(series: pd.Series, low: float = 0.1, high: float = 0.9) -> pd.Series:
    """
    Normaliza una serie al rango [low, high] (útil para gráficos de radar).
    """
    norm = minmax_normalize(series)
    return low + norm * (high - low)