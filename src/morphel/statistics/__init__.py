"""Statistical testing suite for MorpheL experiments.

The paper requires specific paired tests to compare models:
- Exact McNemar for accuracy
- Stratified paired bootstrap for metric differences
- Paired randomization tests
- Holm correction for multiple comparisons

A fixed-checkpoint test with different seeds or regimes does not isolate the
tokenizer effect and must carry an explicit confounding warning.
"""

from __future__ import annotations
