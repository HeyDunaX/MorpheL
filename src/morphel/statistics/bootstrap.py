"""Stratified paired bootstrap for metric differences.

Estimates 95% confidence intervals and p-values for differences in Accuracy
and Macro-F1 between two systems, using stratified sampling to maintain class
balance.
"""

from __future__ import annotations

from typing import Any


def paired_bootstrap(
    gold_labels: Any,
    predictions_a: Any,
    predictions_b: Any,
    samples: int = 10000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Run stratified paired bootstrap resampling.

    Args:
        gold_labels: Array-like of true class indices.
        predictions_a: Array-like of predicted class indices for system A.
        predictions_b: Array-like of predicted class indices for system B.
        samples: Number of bootstrap resamples.
        confidence_level: Desired confidence level for intervals (e.g., 0.95).

    Returns:
        Dictionary containing intervals and empirical p-values.
    """
    raise NotImplementedError("Stub: implement using scipy.stats.bootstrap or custom loop.")
