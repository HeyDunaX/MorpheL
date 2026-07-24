"""Unicode normalization helpers for boundary detection.

Functions in this module operate on individual characters or short strings and
perform no I/O.
"""

from __future__ import annotations

import unicodedata


def is_boundary_punctuation(character: str) -> bool:
    """Return True for Unicode punctuation or symbol characters.

    Uses the Unicode general category: ``P*`` (punctuation) or ``S*`` (symbol).

    Args:
        character: A single Unicode character.

    Returns:
        True if the character is a punctuation or symbol character.
    """
    category = unicodedata.category(character)
    return category.startswith("P") or category.startswith("S")


def strip_boundary_punctuation(token: str) -> str:
    """Strip Unicode punctuation and symbols only at the token boundaries.

    Internal apostrophes and hyphens are preserved. This behaviour makes the
    default tokenizer usable across Latin, Cyrillic, and many other scripts
    while keeping the whitespace-word assumption of the canonical implementation.

    Args:
        token: A whitespace-delimited token string.

    Returns:
        The token with leading and trailing punctuation/symbol characters
        removed. An empty string if the token consists entirely of such
        characters.

    Example:
        >>> strip_boundary_punctuation("(hello,")
        'hello,'
        >>> strip_boundary_punctuation("don't")
        "don't"
    """
    start = 0
    end = len(token)
    while start < end and is_boundary_punctuation(token[start]):
        start += 1
    while end > start and is_boundary_punctuation(token[end - 1]):
        end -= 1
    return token[start:end]
