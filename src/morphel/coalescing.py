"""Dynamic-programming vocabulary coalescing for MorpheL.

After fallback decomposition, the resulting piece list may contain individually
valid vocabulary tokens that can be merged into longer tokens that are also in
the vocabulary. The DP coalescing step finds the minimum-token representation
of the piece sequence under the current vocabulary.

Algorithm
---------
This is a standard interval DP over the piece index:

    best_cost[i] = minimum number of tokens to represent pieces[0:i]
    previous[i]  = start index of the last group in the optimal solution

The inner loop merges pieces[start:end] into a single string and checks
vocabulary membership.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def coalesce_pieces(
    pieces: Sequence[str],
    vocabulary: Mapping[str, int],
) -> list[str]:
    """Find the minimum-token vocabulary-legal coalescing with dynamic programming.

    Given a list of piece strings (each individually in the vocabulary after
    fallback), find the grouping that minimizes the total number of output
    tokens while ensuring every output token is in the vocabulary.

    Args:
        pieces: Sequence of piece strings. May be the output of
            :func:`morphel.fallback.recursive_fallback`.
        vocabulary: Mapping from token string -> integer ID.

    Returns:
        A list of token strings from the vocabulary. The concatenation of the
        returned tokens equals the concatenation of ``pieces``.

    Determinism:
        Fully deterministic given the same vocabulary.

    Note:
        If the DP cannot find a valid solution (which should not occur for
        correctly constructed vocabularies), the function returns the original
        ``pieces`` list unchanged as a safe fallback.
    """
    count = len(pieces)
    if count <= 1:
        return list(pieces)

    infinity = float("inf")
    best_cost = [infinity] * (count + 1)
    previous = [-1] * (count + 1)
    best_cost[0] = 0.0

    for end in range(1, count + 1):
        merged = ""
        for start in range(end - 1, -1, -1):
            merged = pieces[start] + merged
            if merged in vocabulary and best_cost[start] + 1 < best_cost[end]:
                best_cost[end] = best_cost[start] + 1
                previous[end] = start

        if previous[end] == -1 and best_cost[end - 1] + 1 < best_cost[end]:
            best_cost[end] = best_cost[end - 1] + 1
            previous[end] = end - 1

    groups: list[str] = []
    cursor = count
    while cursor > 0:
        start = previous[cursor]
        if start < 0:
            return list(pieces)
        groups.append("".join(pieces[start:cursor]))
        cursor = start
    groups.reverse()
    return groups
