"""Exact McNemar test for accuracy differences.

Computes the exact binomial probability of the discordant pair distribution.
"""

from __future__ import annotations

from typing import Any


def mcnemar_test(
    gold_labels: Any,
    predictions_a: Any,
    predictions_b: Any,
) -> dict[str, Any]:
    """Compute exact McNemar test for paired predictions.

    Calculates the contingency table of discordant pairs (A right/B wrong vs
    A wrong/B right) and computes the exact p-value using the binomial
    distribution.

    Args:
        gold_labels: True class indices.
        predictions_a: Predicted class indices for system A.
        predictions_b: Predicted class indices for system B.

    Returns:
        Dictionary with discordant counts and p-value.
    """
    raise NotImplementedError("Stub: implement using statsmodels.stats.contingency_tables.")
