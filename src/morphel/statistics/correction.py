"""Holm-Bonferroni correction for multiple hypothesis testing.

Applies the step-down Holm method to control the family-wise error rate across
multiple primary metric comparisons.
"""

from __future__ import annotations

from typing import Sequence


def holm_correction(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Apply Holm-Bonferroni correction to a list of p-values.

    Args:
        p_values: List of raw p-values.
        alpha: Target family-wise error rate.

    Returns:
        List of booleans indicating whether each null hypothesis is rejected
        (True = significant).
    """
    raise NotImplementedError("Stub: implement using scipy.stats or custom loop.")
