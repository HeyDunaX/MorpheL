"""Paired randomization tests for arbitrary metrics.

Evaluates the probability of observing the empirical metric difference under
the null hypothesis that system A and system B predictions are exchangeable.
"""

from __future__ import annotations

from typing import Any


def randomization_test(
    gold_labels: Any,
    predictions_a: Any,
    predictions_b: Any,
    metric_fn: Any,
    samples: int = 10000,
) -> dict[str, Any]:
    """Run a paired randomization test.

    For each sample, flips a coin for every example to decide whether to swap
    the prediction from A and B, then computes the metric difference.

    Args:
        gold_labels: True class indices.
        predictions_a: Predicted class indices for system A.
        predictions_b: Predicted class indices for system B.
        metric_fn: Callable taking (gold, pred) and returning a scalar metric.
        samples: Number of randomization samples.

    Returns:
        Dictionary with the empirical p-value.
    """
    raise NotImplementedError("Stub: implement custom swapping loop.")
