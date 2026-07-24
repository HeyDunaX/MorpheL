"""Recursive longest-match fallback for out-of-vocabulary pieces.

When a MorpheL piece string is not in the vocabulary, the fallback decomposes
it by finding the longest vocabulary prefix, then recursively handling the
remainder. If no prefix is found, it tries a vocabulary suffix. As a last
resort, individual characters are returned.

The character-coverage overflow in the vocabulary ensures that single-character
fallback rarely fails for corpus languages, because all corpus characters are
guaranteed to be in the vocabulary.
"""

from __future__ import annotations

from typing import Mapping, Optional

from morphel.config import FallbackStats


def recursive_fallback(
    piece: str,
    vocabulary: Mapping[str, int],
    stats: Optional[FallbackStats] = None,
    min_chunk: int = 1,
) -> list[str]:
    """Re-express an OOV piece using the longest matching vocabulary substrings.

    Algorithm:
    1. If ``piece`` is in the vocabulary, return ``[piece]``.
    2. Otherwise, find the longest prefix of ``piece`` in the vocabulary.
       Recursively handle the remainder.
    3. If no prefix is found, find the longest suffix in the vocabulary.
       Recursively handle the head.
    4. If no substring is found, shatter to individual characters.

    Args:
        piece: The piece string to decompose.
        vocabulary: Mapping from token string -> integer ID.
        stats: Optional :class:`~morphel.config.FallbackStats` accumulator.
            If provided, fallback and shatter events are counted.
        min_chunk: Minimum chunk size before character-level shattering.
            Normally 1 (the default).

    Returns:
        List of token strings from the vocabulary (or individual characters
        in the worst case).

    Determinism:
        Fully deterministic given the same vocabulary.
    """
    if piece in vocabulary:
        return [piece]

    if stats is not None:
        stats.fallback_events += 1

    if len(piece) <= min_chunk:
        if stats is not None:
            stats.char_shatters += 1
        return [piece]

    # Try longest prefix
    for end in range(len(piece), 0, -1):
        prefix = piece[:end]
        if prefix in vocabulary:
            remainder = piece[end:]
            return [prefix] + (
                recursive_fallback(remainder, vocabulary, stats=stats, min_chunk=min_chunk)
                if remainder
                else []
            )

    # Try longest suffix
    for start in range(0, len(piece)):
        suffix = piece[start:]
        if suffix in vocabulary:
            head = piece[:start]
            return (
                recursive_fallback(head, vocabulary, stats=stats, min_chunk=min_chunk)
                if head
                else []
            ) + [suffix]

    # Character-level shattering as last resort
    if stats is not None:
        stats.char_shatters += 1
    return list(piece)
