# Location: src/analysis/lenses/diversity.py
"""
Diversity & Clustering Lens Module (Headless Core).

Groups candidate solutions into clusters (K-Medoids, K-Means, Agglomerative, HDBSCAN)
and tags each solution with its cluster ID ('cluster_str') for discrete visual encoding.
Supports Auto and Manual selection of k, along with Silhouette evaluation.

Pure Python & Pandas module — Framework agnostic.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

# Optional clustering dependencies
try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    KMedoids = None

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    HDBSCAN = None

from src.analysis.lenses.base_analysis import BaseLens
from src.analysis.lenses.utils_metrics import zscore_normalize_dataframe

logger = logging.getLogger(__name__)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def _valid_numeric_metrics(df: pd.DataFrame, metrics: List[str]) -> List[str]:
    """Filters metrics present in DataFrame that are strictly numeric."""
    return [
        m
        for m in metrics
        if m in df.columns and pd.api.types.is_numeric_dtype(df[m])
    ]


def _prepare_matrix(df: pd.DataFrame, metrics: List[str]) -> np.ndarray:
    """Imputes missing values and standardizes features using Z-score scaling (centralized)."""
    x = df[metrics].copy()
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    # Usamos nuestra función centralizada
    return zscore_normalize_dataframe(x, metrics).to_numpy()


def _build_partition_model(
    method: str, k: int
) -> Union[KMeans, AgglomerativeClustering, Any]:
    """Instantiates specified partition clustering model instance."""
    if method == "K-Medoids":
        if KMedoids is not None:
            return KMedoids(n_clusters=k, method="pam", random_state=123)
        logger.warning(
            "scikit-learn-extra KMedoids not installed. Falling back to KMeans."
        )
        return KMeans(n_clusters=k, random_state=123, n_init=10)

    if method == "K-Means":
        return KMeans(n_clusters=k, random_state=123, n_init=10)

    if method == "Agglomerative":
        return AgglomerativeClustering(n_clusters=k)

    return KMeans(n_clusters=k, random_state=123, n_init=10)


def _compute_auto_k(
    x_scaled: np.ndarray, method: str, max_k: int = 10
) -> Tuple[int, Optional[float]]:
    """Determines optimal number of clusters k via silhouette score maximization."""
    n = len(x_scaled)
    if n < 3:
        return 1, None

    best_k = 2
    best_score = -1.0
    upper_k = min(max_k, n - 1)

    for k in range(2, upper_k + 1):
        try:
            model = _build_partition_model(method, k)
            labels = model.fit_predict(x_scaled)
            unique_labels = set(labels)

            if 1 < len(unique_labels) < n:
                score = silhouette_score(x_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        except Exception as err:
            logger.debug("Silhouette evaluation failed for k=%d: %s", k, err)

    return best_k, (best_score if best_score != -1.0 else None)


def _fit_partition_clustering(
    x_scaled: np.ndarray, method: str, k: int
) -> Tuple[np.ndarray, str]:
    """Fits partition clustering model and returns assigned cluster labels."""
    model = _build_partition_model(method, k)
    labels = model.fit_predict(x_scaled)

    method_used = (
        "K-Means fallback"
        if (method == "K-Medoids" and KMedoids is None)
        else method
    )
    return labels, method_used


def _fit_hdbscan(
    x_scaled: np.ndarray, min_cluster_size: int
) -> Tuple[np.ndarray, str]:
    """Fits HDBSCAN density model if available."""
    if HDBSCAN is None:
        logger.warning("HDBSCAN module not installed.")
        labels = np.zeros(len(x_scaled), dtype=int)
        return labels, "HDBSCAN unavailable"

    model = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(x_scaled)
    return labels, "HDBSCAN"


def _fit_agglomerative_distance_cut(
    x_scaled: np.ndarray, distance_threshold: float
) -> Tuple[np.ndarray, str]:
    """Fits Agglomerative clustering cut at fixed distance threshold."""
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        compute_full_tree=True,
    )
    labels = model.fit_predict(x_scaled)
    return labels, "Agglomerative distance cut"


def _compute_silhouette_if_valid(
    x_scaled: np.ndarray, labels: np.ndarray
) -> Optional[float]:
    """Safely calculates silhouette score if valid number of clusters exist."""
    unique_labels = set(labels) - {-1}
    n = len(labels)

    if len(unique_labels) <= 1 or len(unique_labels) >= n:
        return None

    try:
        return float(silhouette_score(x_scaled, labels))
    except Exception:
        return None


def _add_cluster_labels(
    result: pd.DataFrame,
    labels: np.ndarray,
    method_used: str,
    metrics_used: List[str],
    silhouette: Optional[float] = None,
) -> pd.DataFrame:
    """Attaches cluster IDs, labels, sizes, and metadata to output DataFrame."""
    res = result.copy()
    res["cluster"] = labels
    res["cluster_str"] = res["cluster"].astype(str).replace("-1", "Noise")

    id_col = "id" if "id" in res.columns else res.index
    cluster_sizes = res.groupby("cluster_str")[id_col].transform("size")
    res["group_label"] = (
        "Cluster " + res["cluster_str"] + " (n=" + cluster_sizes.astype(str) + ")"
    )

    n_clusters = (
        res["cluster"]
        .dropna()
        .astype(int)
        .loc[lambda v: v != -1]
        .nunique()
    )
    noise_count = int(res["cluster"].eq(-1).sum())

    res["diversity_method"] = method_used
    res["diversity_metrics"] = ", ".join(metrics_used)
    res["diversity_n_clusters"] = n_clusters
    res["diversity_noise_count"] = noise_count

    if silhouette is not None:
        res["diversity_silhouette"] = silhouette
        res.attrs["silhouette_score"] = silhouette

    return res


# =====================================================
# MAIN LENS CLASS
# =====================================================

class DiversityLens(BaseLens):
    """Lente de Diversidad: Agrupa soluciones en clusters con soporte para HDBSCAN y Agglomerative avanzado."""
    name: str = "Diversity"
    category: str = "Exploration"
    description: str = "Groups solutions into clusters to explore diversity and representative subsets."

    def get_schema(
        self,
        df: pd.DataFrame,
        dimensions: List[str],
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Devuelve el esquema de parámetros que la UI debe renderizar de forma agnóstica.
        Inspecciona 'context' (que contiene los parámetros actuales) para decidir
        qué controles mostrar.
        """
        if df is None or df.empty or not dimensions:
            return []

        # Filtrar dimensiones numéricas
        filtered_dimensions = [
            d for d in dimensions
            if not d.startswith(("req_", "x_", "var_", "item_"))
            and d.lower() not in ("id", "selected_ids", "selected_ids_str", "is_feasible")
        ]

        if not filtered_dimensions:
            return []

        safe_context = context if context is not None else {}
        current_method = safe_context.get("method", "K-Medoids")

        n = len(df)
        max_k = min(15, max(2, n - 1)) if n > 2 else 5

        schema: List[Dict[str, Any]] = []

        # --- 1. Método y Métricas (Siempre visibles) ---
        schema.append({
            "key": "method",
            "label": "Clustering Method",
            "type": "select",
            "options": ["K-Medoids", "K-Means", "Agglomerative", "HDBSCAN"],
            "default": "K-Medoids",
        })

        schema.append({
            "key": "cluster_metrics",
            "label": "Metrics for Clustering",
            "type": "multiselect",
            "options": filtered_dimensions,
            "default": filtered_dimensions,
        })

        # --- 2. Controles específicos para métodos particionales (K-Medoids, K-Means) ---
        if current_method in ["K-Medoids", "K-Means"]:
            # Calcular k óptimo para sugerir
            metrics = safe_context.get("cluster_metrics", filtered_dimensions)
            valid_metrics = _valid_numeric_metrics(df, metrics)
            if len(valid_metrics) >= 2:
                x_scaled = _prepare_matrix(df, valid_metrics)
                k_opt, sil_opt = _compute_auto_k(x_scaled, current_method, max_k=max_k)
                if k_opt is not None:
                    schema.append({
                        "key": "manual_k_info",
                        "label": "Silhouette Suggestion",
                        "type": "info",
                        "content": f"**Suggested k = {k_opt}** (Silhouette: {sil_opt:.3f})",
                    })
                    default_k = k_opt
                else:
                    default_k = min(3, max_k)
            else:
                default_k = min(3, max_k)

            schema.append({
                "key": "k",
                "label": "k Groups",
                "type": "slider",
                "min": 2,
                "max": max_k,
                "default": default_k,
            })

        # --- 3. Controles específicos para Agglomerative ---
        elif current_method == "Agglomerative":
            schema.append({
                "key": "agglomerative_mode",
                "label": "Hierarchy Cut Mode",
                "type": "select",
                "options": ["Number of Groups", "Distance Cut"],
                "default": "Number of Groups",
            })

            current_agg_mode = safe_context.get("agglomerative_mode", "Number of Groups")
            if current_agg_mode == "Number of Groups":
                # Similar sugerencia para Agglomerative
                metrics = safe_context.get("cluster_metrics", filtered_dimensions)
                valid_metrics = _valid_numeric_metrics(df, metrics)
                if len(valid_metrics) >= 2:
                    x_scaled = _prepare_matrix(df, valid_metrics)
                    k_opt, sil_opt = _compute_auto_k(x_scaled, "Agglomerative", max_k=max_k)
                    if k_opt is not None:
                        schema.append({
                            "key": "manual_k_info",
                            "label": "Silhouette Suggestion",
                            "type": "info",
                            "content": f"**Suggested k = {k_opt}** (Silhouette: {sil_opt:.3f})",
                        })
                        default_k = k_opt
                    else:
                        default_k = min(3, max_k)
                else:
                    default_k = min(3, max_k)

                schema.append({
                    "key": "k",
                    "label": "k Groups",
                    "type": "slider",
                    "min": 2,
                    "max": max_k,
                    "default": default_k,
                })
            else:
                schema.append({
                    "key": "distance_threshold",
                    "label": "Distance Cutoff",
                    "type": "slider_float",
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "default": 2.0,
                })

        # --- 4. Controles específicos para HDBSCAN ---
        elif current_method == "HDBSCAN":
            schema.append({
                "key": "cluster_size_mode",
                "label": "Cluster Size",
                "type": "select",
                "options": ["Auto", "Manual"],
                "default": "Auto",
            })

            current_size_mode = safe_context.get("cluster_size_mode", "Auto")
            if current_size_mode == "Auto":
                schema.append({
                    "key": "granularity",
                    "label": "Cluster Granularity",
                    "type": "select",
                    "options": ["Small (~5%)", "Medium (~10%)", "Large (~20%)"],
                    "default": "Medium (~10%)",
                })
            else:
                schema.append({
                    "key": "min_cluster_size",
                    "label": "Minimum Cluster Size",
                    "type": "slider",
                    "min": 2,
                    "max": max(2, n),
                    "default": max(2, int(0.1 * n)),
                })

            schema.append({
                "key": "exclude_noise",
                "label": "Exclude noise solutions",
                "type": "checkbox",
                "default": True,
            })

        return schema

    def apply(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Aplica el clustering seleccionado y filtra por el grupo elegido.
        """
        if df is None or df.empty or len(df) < 2:
            return df

        result = df.copy()
        metrics_from_context = (context or {}).get("metrics", []) + (context or {}).get("selected_indicators", [])
        method = params.get("method", "K-Medoids")
        cluster_metrics = params.get("cluster_metrics", metrics_from_context)
        cluster_metrics = _valid_numeric_metrics(result, cluster_metrics)

        if len(cluster_metrics) < 2:
            return result

        x_scaled = _prepare_matrix(result, cluster_metrics)
        silhouette: Optional[float] = None

        # 1. Ejecutar clustering según método
        if method in ["K-Medoids", "K-Means"]:
            # Ahora siempre se usa el valor del slider (k)
            k = max(2, min(params.get("k", 2), len(result)))
            model = _build_partition_model(method, k)
            labels = model.fit_predict(x_scaled)
            silhouette = _compute_silhouette_if_valid(x_scaled, labels)
            method_used = method

            result = _add_cluster_labels(result, labels, method_used, cluster_metrics, silhouette)
            result["diversity_k"] = k

        elif method == "Agglomerative":
            agglomerative_mode = params.get("agglomerative_mode", "Number of Groups")

            if agglomerative_mode == "Distance Cut":
                dist_thresh = params.get("distance_threshold", 2.0)
                labels, method_used = _fit_agglomerative_distance_cut(x_scaled, dist_thresh)
                silhouette = _compute_silhouette_if_valid(x_scaled, labels)
                result = _add_cluster_labels(result, labels, method_used, cluster_metrics, silhouette)
                result["diversity_distance_threshold"] = dist_thresh
            else:
                # Ahora siempre se usa el valor del slider (k)
                k = max(2, min(params.get("k", 2), len(result)))
                model = _build_partition_model(method, k)
                labels = model.fit_predict(x_scaled)
                silhouette = _compute_silhouette_if_valid(x_scaled, labels)
                method_used = method

                result = _add_cluster_labels(result, labels, method_used, cluster_metrics, silhouette)
                result["diversity_k"] = k

        elif method == "HDBSCAN":
            n = len(result)
            size_mode = params.get("cluster_size_mode", "Auto")

            if size_mode == "Manual":
                min_cluster_size = params.get("min_cluster_size", max(2, int(0.1 * n)))
            else:
                granularity = params.get("granularity", "Medium (~10%)")
                if granularity == "Small (~5%)":
                    min_cluster_size = max(2, int(0.05 * n))
                elif granularity == "Large (~20%)":
                    min_cluster_size = max(2, int(0.20 * n))
                else:
                    min_cluster_size = max(2, int(0.10 * n))

            labels, method_used = _fit_hdbscan(x_scaled, min_cluster_size)
            silhouette = _compute_silhouette_if_valid(x_scaled, labels)
            result = _add_cluster_labels(result, labels, method_used, cluster_metrics, silhouette)
            result["diversity_min_cluster_size"] = min_cluster_size

            if params.get("exclude_noise", True) and not params.get("selected_cluster"):
                result = result[result["cluster"] != -1].copy()


        # 2. Filtrar por cluster seleccionado (SOI)
        selected_cluster = params.get("selected_cluster", "All")
        if selected_cluster and selected_cluster != "All" and "cluster_str" in result.columns:
            result = result[result["cluster_str"] == str(selected_cluster)].copy()

        return result


# ===================================================================
# EXPOSICIÓN A NIVEL DE MÓDULO (Para compatibilidad con el registro)
# ===================================================================

_lens_instance = DiversityLens()


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