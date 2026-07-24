"""Gumbel-max sampling and random seed management for MorpheL.

The Gumbel-max trick samples a categorical distribution by adding i.i.d.
Gumbel noise to log-logits and taking the argmax. At temperature T=0 the
sampling reduces to deterministic argmax, which is used for vocabulary
induction and evaluation.

The canonical temperature schedule is:
- T=1.0 during stochastic induction (default training temperature).
- T=0.0 for the deterministic segmentation cache used in vocabulary induction
  and all evaluation. This is an exact argmax, not a low-temperature
  approximation.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python built-in random and NumPy for reproducible Gumbel sampling.

    This function must be called once before vocabulary induction begins.
    It seeds both the Python ``random`` module and ``numpy.random``, matching
    the canonical implementation.

    Args:
        seed: Integer seed value. The same seed produces the same Gumbel
            samples and therefore the same stochastic segmentation trajectory.
    """
    random.seed(seed)
    np.random.seed(seed)


def gumbel_sample(logits: Sequence[float], temperature: float) -> int:
    """Sample an index with the Gumbel-max trick.

    A non-positive temperature returns the deterministic argmax used for
    vocabulary induction and evaluation. This is the canonical MorpheL
    sampling procedure.

    Args:
        logits: Sequence of real-valued logits (one per class). Must be
            non-empty.
        temperature: Sampling temperature. Values <= 0.0 trigger deterministic
            argmax (T=0 mode).

    Returns:
        The sampled (or argmax) index.

    Raises:
        ValueError: If ``logits`` is empty.

    Determinism:
        At temperature <= 0.0 this function is deterministic.
        At temperature > 0.0 the result depends on the NumPy random state.
        Seed NumPy with :func:`set_seed` before calling for reproducibility.
    """
    if not logits:
        raise ValueError("logits must be non-empty.")
    if temperature <= 0.0:
        return int(np.argmax(np.asarray(logits, dtype=np.float64)))

    scores = np.asarray(logits, dtype=np.float64)
    uniform = np.random.uniform(1e-20, 1.0, len(scores))
    gumbel_noise = -np.log(-np.log(uniform))
    return int(np.argmax(scores / temperature + gumbel_noise))
