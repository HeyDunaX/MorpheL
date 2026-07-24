"""Regime C downstream adaptation.

Contract
--------
- Replacement embeddings trainable.
- Entire encoder trainable.
- Classifier trainable.
- No `torch.no_grad`.
- No detached hidden states.
- Zero frozen parameters.
"""

from __future__ import annotations

from typing import Any


def configure_regime_c(model: Any) -> Any:
    """Apply Regime C trainable/frozen constraints to the model.

    Args:
        model: PyTorch model instance.

    Returns:
        The model with `requires_grad` set to True for all parameters.
    """
    raise NotImplementedError("Stub: pending project owner implementation.")
