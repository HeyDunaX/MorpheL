"""Checkpoint selection and per-epoch evaluation.

Fairness rule: checkpoints must be selected using validation accuracy only.
Test labels must not be seen during training.
"""

from __future__ import annotations

from typing import Any


def get_best_checkpoint(trainer_state: Any, metric: str = "eval_accuracy") -> Any:
    """Identify the best checkpoint based on the target validation metric.

    Args:
        trainer_state: State object containing epoch history.
        metric: The metric to maximize (e.g., "eval_accuracy").

    Returns:
        Checkpoint path or metadata.
    """
    raise NotImplementedError("Stub: pending project owner implementation.")
