"""Regime B downstream adaptation.

Contract
--------
- Replacement word-embedding matrix trainable.
- Classifier trainable.
- All remaining encoder parameters frozen (encoder layers, layer norm,
  position embeddings).
- Do not use `torch.no_grad` around the encoder forward pass because gradients
  must reach the word embeddings.
- Exact trainable parameter scope must be asserted.
"""

from __future__ import annotations

from typing import Any


def configure_regime_b(model: Any) -> Any:
    """Apply Regime B trainable/frozen constraints to the model.

    Args:
        model: PyTorch model instance.

    Returns:
        The model with `requires_grad` set appropriately.
    """
    raise NotImplementedError("Stub: pending project owner implementation.")
