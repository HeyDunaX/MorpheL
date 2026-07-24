"""Shared type aliases for MorpheL.

Centralizing these prevents circular imports and documents the semantic
meaning of recurring compound types.
"""

from __future__ import annotations

from typing import Callable

# A sequence of (boundary_position, mi_score) pairs for one word.
# Positions are byte-offset-free character indices within the word string.
Boundary = tuple[int, float]

# Mapping from word string to its list of candidate boundaries.
MIIndex = dict[str, list[Boundary]]

# Mapping from word string to its list of MorpheL piece strings.
SegmentationCache = dict[str, list[str]]

# A callable that splits a raw text string into orthographic word strings.
# Researchers must supply this for languages without reliable whitespace
# word boundaries (Chinese, Japanese, Thai, …).
#
# The same callable must be used consistently across:
#   tokenizer induction, intrinsic evaluation, Regime A, Regime B, Regime C,
#   validation, test evaluation, and significance testing.
TokenizerFunction = Callable[[str], list[str]]
