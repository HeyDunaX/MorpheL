"""Regime A downstream adaptation.

Contract
--------
- Replacement word embeddings initialized with the declared mapping procedure.
- Word embeddings frozen.
- Encoder frozen.
- Classification head trainable.
- Exactly 2,307 trainable parameters for a 768-by-3 linear head with bias
  (assuming XLM-R base).
- No encoder gradient graph required (inputs can be detached or `torch.no_grad`
  can be used around the encoder).
"""

from __future__ import annotations

from typing import Any


def configure_regime_a(model: Any) -> Any:
    """Apply Regime A trainable/frozen constraints to the model.

    Args:
        model: PyTorch model instance.

    Returns:
        The model with `requires_grad` set appropriately.
    """
    raise NotImplementedError("Stub: pending project owner implementation.")
