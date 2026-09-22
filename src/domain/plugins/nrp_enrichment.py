# Location: src/domain/plugins/nrp_enrichment.py
"""
NRP Enrichment Utilities.

This module provides the NRPEnricher class, which computes a set of
domain‑specific enrichment indicators for solutions in the Next Release
Problem (NRP) domain. Indicators include productivity, effectiveness,
squandering, robustness, and others, derived from the solution's
objective attributes and decision variables.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, Iterable, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
EPS: float = 1e-9  # Small epsilon to avoid division by zero


@dataclass
class IndicatorDefinition:
    """
    Definition of an enrichment indicator.

    Attributes:
        name: Unique identifier for the indicator.
        description: Human‑readable description.
        required_attributes: List of attribute names that must be present
            in the DataFrame for the indicator to be computable.
    """
    name: str
    description: str
    required_attributes: List[str] = field(default_factory=list)


class NRPEnricher:
    """
    Calculator of enrichment indicators for NRP solutions.

    This class provides a registry of available indicators and a method to
    compute them on a DataFrame of solutions. Each indicator is a function
    of the solution's attributes (e.g., satisfaction, effort, cost).
    """

    INDICATORS_REGISTRY: Dict[str, IndicatorDefinition] = {
        "scope": IndicatorDefinition(
            "scope",
            "Ratio of included requirements.",
            []
        ),
        "productivity": IndicatorDefinition(
            "productivity",
            "Satisfaction per unit of effort.",
            ["satisfaction", "effort"]
        ),
        "effectiveness": IndicatorDefinition(
            "effectiveness",
            "Satisfaction per unit of financial cost.",
            ["satisfaction", "cost"]
        ),
        "squandering": IndicatorDefinition(
            "squandering",
            "Wasted capacity relative to maximum effort.",
            ["effort"]
        ),
        "dirtiness": IndicatorDefinition(
            "dirtiness",
            "Dissatisfaction per unit of effort.",
            ["dissatisfaction", "effort"]
        ),
        "annoyance": IndicatorDefinition(
            "annoyance",
            "Dissatisfaction relative to satisfaction.",
            ["dissatisfaction", "satisfaction"]
        ),
        "stickiness": IndicatorDefinition(
            "stickiness",
            "Prevalence per unit of effort.",
            ["prevalence", "effort"]
        ),
        "robustness": IndicatorDefinition(
            "robustness",
            "Satisfaction versus instability.",
            ["satisfaction", "instability"]
        ),
        "fragility": IndicatorDefinition(
            "fragility",
            "Failure risk due to prevalence and instability.",
            ["prevalence", "instability", "effort"]
        ),
        "response": IndicatorDefinition(
            "response",
            "Technical response speed (effort per time).",
            ["time", "effort"]
        ),
        "opportunity": IndicatorDefinition(
            "opportunity",
            "Time usage versus satisfaction.",
            ["satisfaction", "time"]
        ),
        "usage_efficiency": IndicatorDefinition(
            "usage_efficiency",
            "Prevalence per unit of financial cost.",
            ["prevalence", "cost"]
        ),
    }

    @classmethod
    def get_calculated_indicators(cls) -> Dict[str, IndicatorDefinition]:
        """
        Return the registry of available indicators.

        Returns:
            Dictionary mapping indicator names to their definitions.
        """
        return cls.INDICATORS_REGISTRY

    @classmethod
    def compute_indicators(
        cls,
        df: pd.DataFrame,
        indicators: Iterable[str],
        decision_var_prefix: str = "req_"
    ) -> pd.DataFrame:
        """
        Compute the requested enrichment indicators on the given DataFrame.

        Each indicator is calculated column‑wise using the existing attributes.
        Missing required attributes will cause the indicator to be skipped
        with a warning.

        Args:
            df: Input DataFrame containing solutions.
            indicators: Iterable of indicator names to compute.
            decision_var_prefix: Prefix used for decision variable columns
                (used to compute the 'scope' indicator).

        Returns:
            A new DataFrame with the computed indicators added as columns.
        """
        if df is None or df.empty:
            return df

        result = df.copy()
        req_cols = [c for c in result.columns if c.startswith(decision_var_prefix)]

        for indicator in indicators:
            try:
                if indicator == "productivity":
                    result[indicator] = result["satisfaction"] / np.maximum(result["effort"], EPS)

                elif indicator == "effectiveness":
                    result[indicator] = result["satisfaction"] / np.maximum(result["cost"], EPS)

                elif indicator == "squandering":
                    effort_max = result["effort"].max()
                    result[indicator] = (effort_max - result["effort"]) / np.maximum(effort_max, EPS)

                elif indicator == "dirtiness":
                    result[indicator] = np.where(
                        result["dissatisfaction"] == 0, 0.0,
                        result["dissatisfaction"] / np.maximum(result["effort"], EPS)
                    )

                elif indicator == "annoyance":
                    result[indicator] = np.where(
                        result["dissatisfaction"] == 0, 0.0,
                        result["dissatisfaction"] / np.maximum(result["satisfaction"], EPS)
                    )

                elif indicator == "stickiness":
                    result[indicator] = result["prevalence"] / np.maximum(result["effort"], EPS)

                elif indicator == "robustness":
                    result[indicator] = result["satisfaction"] / np.maximum(result["instability"], EPS)

                elif indicator == "fragility":
                    result[indicator] = (result["prevalence"] * result["instability"]) / np.maximum(result["effort"], EPS)

                elif indicator == "response":
                    result[indicator] = np.where(
                        result["time"] == 0, 0.0,
                        result["effort"] / np.maximum(result["time"], EPS)
                    )

                elif indicator == "opportunity":
                    result[indicator] = np.where(
                        result["satisfaction"] == 0, 0.0,
                        result["satisfaction"] / np.maximum(result["time"], EPS)
                    )

                elif indicator == "usage_efficiency":
                    result[indicator] = (result["prevalence"] / result["cost"].replace(0, np.nan)).fillna(0.0)

                elif indicator == "scope":
                    if req_cols:
                        result[indicator] = result[req_cols].sum(axis=1) / len(req_cols)

            except Exception as exc:
                logger.warning("[NRPEnricher] Could not compute indicator '%s': %s", indicator, exc)

        return result