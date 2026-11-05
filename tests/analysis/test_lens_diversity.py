# tests/test_lens_diversity.py
"""
Unit tests for the Diversity lens (clustering).

These tests verify that the Diversity lens correctly performs clustering
using various methods (K‑Means, K‑Medoids, Agglomerative, HDBSCAN),
handles automatic and manual parameter selection, and filters results
by cluster or noise exclusion.
"""

import pytest
import pandas as pd
import numpy as np

from src.analysis.lenses.diversity import DiversityLens


@pytest.fixture
def diversity_df():
    """Create a sample DataFrame with 10 solutions and 4 numeric metrics."""
    np.random.seed(42)
    data = {
        "id": [f"s{i}" for i in range(10)],
        "cost": np.random.randint(10, 50, 10),
        "effort": np.random.randint(5, 20, 10),
        "satisfaction": np.random.randint(20, 100, 10),
        "risk": np.random.randint(1, 10, 10),
    }
    return pd.DataFrame(data)


def test_diversity_lens_kmeans_basic(diversity_df):
    """
    Test that K‑Means clustering assigns cluster labels correctly.

    The result should contain cluster columns and have exactly 3 clusters.
    """
    lens = DiversityLens()
    params = {
        "method": "K-Means",
        "cluster_metrics": ["cost", "effort", "satisfaction"],
        "k": 3,
    }
    result = lens.apply(diversity_df, params)
    assert "cluster" in result.columns
    assert "cluster_str" in result.columns
    assert "group_label" in result.columns
    assert result["cluster"].nunique() == 3
    assert len(result) == len(diversity_df)


def test_diversity_lens_kmedoids_fallback(diversity_df):
    """
    Test that K‑Medoids clustering works (or falls back to K‑Means if not installed).

    The result should have cluster assignments and a 'diversity_method' column.
    """
    lens = DiversityLens()
    params = {
        "method": "K-Medoids",
        "cluster_metrics": ["cost", "effort", "satisfaction"],
        "k": 3,
    }
    result = lens.apply(diversity_df, params)
    assert "cluster" in result.columns
    assert result["cluster"].nunique() == 3
    assert "diversity_method" in result.columns


def test_diversity_lens_agglomerative(diversity_df):
    """
    Test Agglomerative clustering with a fixed number of groups.
    """
    lens = DiversityLens()
    params = {
        "method": "Agglomerative",
        "agglomerative_mode": "Number of Groups",
        "cluster_metrics": ["cost", "effort", "satisfaction"],
        "k": 3,
    }
    result = lens.apply(diversity_df, params)
    assert result["cluster"].nunique() == 3
    assert "diversity_k" in result.columns


def test_diversity_lens_agglomerative_distance_cut(diversity_df):
    """
    Test Agglomerative clustering with a distance cut‑off.
    """
    lens = DiversityLens()
    params = {
        "method": "Agglomerative",
        "agglomerative_mode": "Distance Cut",
        "cluster_metrics": ["cost", "effort", "satisfaction"],
        "distance_threshold": 1.5,
    }
    result = lens.apply(diversity_df, params)
    assert "cluster" in result.columns
    assert "diversity_distance_threshold" in result.columns
    assert result["diversity_distance_threshold"].iloc[0] == 1.5


def test_diversity_lens_hdbscan_auto(diversity_df):
    """
    Test HDBSCAN clustering in automatic mode.
    Skip if HDBSCAN is not installed.
    """
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        pytest.skip("HDBSCAN not installed")
    lens = DiversityLens()
    params = {
        "method": "HDBSCAN",
        "cluster_size_mode": "Auto",
        "granularity": "Medium (~10%)",
        "cluster_metrics": ["cost", "effort", "satisfaction"],
        "exclude_noise": False,
    }
    result = lens.apply(diversity_df, params)
    assert "cluster" in result.columns
    assert "diversity_method" in result.columns
    assert result["diversity_method"].iloc[0] == "HDBSCAN"


def test_diversity_lens_hdbscan_manual(diversity_df):
    """
    Test HDBSCAN clustering in manual mode with a specified minimum cluster size.
    Skip if HDBSCAN is not installed.
    """
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        pytest.skip("HDBSCAN not installed")
    lens = DiversityLens()
    params = {
        "method": "HDBSCAN",
        "cluster_size_mode": "Manual",
        "min_cluster_size": 2,
        "cluster_metrics": ["cost", "effort", "satisfaction"],
        "exclude_noise": False,
    }
    result = lens.apply(diversity_df, params)
    assert "cluster" in result.columns
    assert "diversity_min_cluster_size" in result.columns


def test_diversity_lens_exclude_noise(diversity_df):
    """
    Test that the 'exclude_noise' option removes noise points.

    With min_cluster_size=5, all points are expected to be noise,
    so the filtered DataFrame should be empty.
    """
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        pytest.skip("HDBSCAN not installed")
    lens = DiversityLens()
    params = {
        "method": "HDBSCAN",
        "cluster_size_mode": "Manual",
        "min_cluster_size": 5,
        "cluster_metrics": ["cost", "effort", "satisfaction"],
        "exclude_noise": True,
    }
    result = lens.apply(diversity_df, params)
    # With min_cluster_size=5 all points are noise, so the result should be empty
    assert result.empty, "DataFrame should be empty because all points are noise"


def test_diversity_lens_filter_by_cluster(diversity_df):
    """
    Test filtering the result to keep only solutions from a specific cluster.
    """
    lens = DiversityLens()
    params = {
        "method": "K-Means",
        "cluster_metrics": ["cost", "effort", "satisfaction"],
        "k": 3,
        "selected_cluster": "0",
    }
    result = lens.apply(diversity_df, params)
    assert all(result["cluster_str"] == "0")


def test_diversity_lens_insufficient_metrics(diversity_df):
    """
    Test that the lens returns the original DataFrame unchanged when fewer than
    two numeric metrics are provided.
    """
    df_single = diversity_df[["id", "cost"]].copy()
    lens = DiversityLens()
    params = {
        "method": "K-Means",
        "cluster_metrics": ["cost"],
        "k": 3,
    }
    result = lens.apply(df_single, params)
    pd.testing.assert_frame_equal(result, df_single)


def test_diversity_lens_schema(diversity_df):
    """
    Test that the lens schema contains the expected controls.
    """
    lens = DiversityLens()
    schema = lens.get_schema(
        diversity_df,
        ["cost", "effort", "satisfaction", "risk"],
        context={"method": "K-Means"}
    )
    assert isinstance(schema, list)
    assert len(schema) > 0
    keys = [item["key"] for item in schema]
    assert "method" in keys
    assert "cluster_metrics" in keys