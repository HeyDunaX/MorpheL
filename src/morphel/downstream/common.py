"""Shared utilities for downstream training.

Contains common training loop components, metrics aggregation, and data
collators.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_classification_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute standard text classification metrics.

    Args:
        eval_pred: Output of the HF Trainer evaluation loop (logits, labels).

    Returns:
        Dictionary of metrics including accuracy and macro-F1.
    """
    raise NotImplementedError("Stub: pending project owner implementation.")
