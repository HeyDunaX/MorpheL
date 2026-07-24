"""Global prefix/suffix mutual-information index for MorpheL.

The MI index is computed over the **entire corpus** (not per-sentence).
Token-frequency weighting ensures that frequent word types contribute more to
the prefix and suffix counts.

MI formula
----------
For word ``w`` split at position ``i``:

    MI(w, i) = log(P(w) / (P(w[:i]) * P(w[i:])) + 1e-9)

where all probabilities are relative frequencies over the full token corpus.
The ``+ 1e-9`` term inside the log provides numerical stability; it is part of
the canonical formula and must not be removed.

Only boundaries with MI > ``mi_threshold`` are retained (default 0.0).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Mapping

from tqdm.auto import tqdm

from morphel.boundaries import is_plausible_boundary
from morphel.logging_utils import get_logger
from morphel.typing import Boundary, MIIndex

logger = get_logger(__name__)


def build_global_mi_index(
    word_frequency: Mapping[str, int],
    vowels: set[str],
    mi_threshold: float = 0.0,
    use_vowel_boundary_filter: bool = True,
) -> MIIndex:
    """Build the corpus-global prefix/suffix MI index.

    This is a two-pass algorithm:
    1. Accumulate prefix and suffix frequencies over all plausible boundary
       positions, weighted by the word's corpus-token frequency.
    2. Compute MI for each (word, position) pair and retain entries with
       MI > mi_threshold.

    Args:
        word_frequency: Mapping from word string to corpus-token frequency.
        vowels: Language-specific vowel inventory. See :mod:`morphel.boundaries`
            for the language assumption.
        mi_threshold: Minimum MI score for a boundary to be retained.
            The default 0.0 keeps all positive-MI boundaries.
        use_vowel_boundary_filter: Whether to apply the vowel-transition
            heuristic. See :func:`morphel.boundaries.is_plausible_boundary`.

    Returns:
        A :class:`~morphel.typing.MIIndex` mapping each word string to its
        list of ``(position, mi_score)`` pairs, sorted by descending MI score.
        Words with no plausible positive-MI boundary are excluded.

    Raises:
        ValueError: If the corpus is empty after tokenization.

    Determinism:
        This function is fully deterministic given the same inputs.

    Note:
        MI is computed at the corpus level. A word type that appears only once
        in the corpus will have MI values influenced by the global prefix/suffix
        distributions, not just its own occurrence.
    """
    total_tokens = sum(int(count) for count in word_frequency.values())
    if total_tokens <= 0:
        raise ValueError("The corpus is empty after tokenization.")

    # Pass 1: accumulate prefix and suffix frequencies
    prefix_frequency: Counter = Counter()
    suffix_frequency: Counter = Counter()

    for word, count in tqdm(
        word_frequency.items(),
        desc="Counting prefix/suffix frequencies",
    ):
        if len(word) <= 2:
            continue
        for position in range(2, len(word) - 1):
            if not is_plausible_boundary(
                word, position, vowels, use_filter=use_vowel_boundary_filter
            ):
                continue
            prefix_frequency[word[:position]] += count
            suffix_frequency[word[position:]] += count

    # Pass 2: compute MI for each word and position
    mi_index: MIIndex = {}
    boundary_count = 0

    for word, word_count in tqdm(
        word_frequency.items(),
        desc="Computing global MI index",
    ):
        entries: list[Boundary] = []
        if len(word) <= 2:
            continue

        for position in range(2, len(word) - 1):
            if not is_plausible_boundary(
                word, position, vowels, use_filter=use_vowel_boundary_filter
            ):
                continue

            prefix_count = prefix_frequency[word[:position]]
            suffix_count = suffix_frequency[word[position:]]
            if prefix_count == 0 or suffix_count == 0:
                continue

            word_probability = word_count / total_tokens
            prefix_probability = prefix_count / total_tokens
            suffix_probability = suffix_count / total_tokens

            # Canonical MI formula — the 1e-9 term is inside log() for stability
            mi_score = math.log(
                word_probability / (prefix_probability * suffix_probability) + 1e-9
            )

            if mi_score > mi_threshold:
                entries.append((position, float(mi_score)))

        if entries:
            mi_index[word] = entries
            boundary_count += len(entries)

    logger.info(
        "Global MI index: %d positive boundaries across %d word types",
        boundary_count,
        len(mi_index),
    )
    return mi_index
