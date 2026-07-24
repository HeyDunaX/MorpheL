"""Vocabulary construction for MorpheL.

The nominal vocabulary contains:
1. Special tokens in XLM-R-compatible order.
2. The most frequent contiguous spans from MorpheL segmentations, up to the
   nominal ``vocab_size`` budget.
3. Missing corpus characters appended as coverage-only overflow entries.

The overflow step ensures that every corpus character can be represented,
matching the canonical source notebook behavior.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from tqdm.auto import tqdm

from morphel.constants import SPECIAL_TOKENS
from morphel.logging_utils import get_logger
from morphel.typing import SegmentationCache

logger = get_logger(__name__)


def count_contiguous_spans(
    word_frequency: Mapping[str, int],
    segmentation_cache: SegmentationCache,
) -> Counter:
    """Count all contiguous spans permitted by the cached MorpheL segmentations.

    For each word type, the segmentation cache provides the T=0 piece list.
    Every contiguous sub-sequence of pieces forms a candidate subword span.
    Counts are weighted by the word type's corpus-token frequency.

    Args:
        word_frequency: Mapping from word string to corpus-token frequency.
        segmentation_cache: Deterministic T=0 segmentation cache.

    Returns:
        A :class:`collections.Counter` mapping span string -> weighted count.

    Determinism:
        Fully deterministic given the same inputs.
    """
    frequencies: Counter = Counter()
    for word, count in tqdm(
        word_frequency.items(),
        desc="Counting contiguous MorpheL spans",
    ):
        pieces = segmentation_cache.get(word, [word])
        piece_count = len(pieces)
        for start in range(piece_count):
            for end in range(start + 1, piece_count + 1):
                frequencies["".join(pieces[start:end])] += count
    return frequencies


def build_character_alphabet(word_types: Iterable[str]) -> set[str]:
    """Return all unique characters found in the corpus word types.

    Used to construct the character-coverage overflow entries appended after
    the nominal vocabulary budget.

    Args:
        word_types: Iterable of word strings.

    Returns:
        Set of all unique characters found across all word types.

    Determinism:
        Fully deterministic.
    """
    alphabet: set[str] = set()
    for word in word_types:
        alphabet.update(word)
    return alphabet


def build_vocabulary(
    subword_frequency: Counter,
    character_alphabet: set[str],
    vocab_size: int,
    min_frequency: int,
    special_tokens: Sequence[str] = SPECIAL_TOKENS,
) -> tuple[dict[str, int], dict[str, int]]:
    """Build the nominal subword vocabulary plus character-coverage overflow.

    The nominal ``vocab_size`` includes special tokens and high-frequency
    MorpheL spans. Missing corpus characters are appended in sorted order as
    coverage-only overflow entries, matching the canonical source notebook.

    Special-token IDs are assigned first in the declared order:
        <s>=0, <pad>=1, </s>=2, <unk>=3, <mask>=4

    Args:
        subword_frequency: Counter of span strings from :func:`count_contiguous_spans`.
        character_alphabet: Set of all corpus characters from
            :func:`build_character_alphabet`.
        vocab_size: Target nominal vocabulary size (includes special tokens).
        min_frequency: Minimum corpus frequency for a span to enter the
            nominal vocabulary.
        special_tokens: Sequence of special token strings in insertion order.
            Defaults to the XLM-R-compatible order.

    Returns:
        A tuple of:
        - ``vocabulary``: Mapping from token string -> integer ID.
        - ``diagnostics``: Dict with counts of nominal, added, and overflow
          tokens.

    Determinism:
        Fully deterministic. Vocabulary IDs are assigned by descending
        frequency rank, then alphabetically for ties (via ``most_common()``
        which is insertion-ordered in CPython for tied counts; this is
        deterministic when the Counter is built from a deterministic process).
    """
    vocabulary: dict[str, int] = {
        token: index for index, token in enumerate(special_tokens)
    }

    added_pieces = 0
    for token, frequency in subword_frequency.most_common():
        if len(vocabulary) >= vocab_size:
            break
        if frequency < min_frequency or token in vocabulary:
            continue
        vocabulary[token] = len(vocabulary)
        added_pieces += 1

    overflow_characters = 0
    for character in sorted(character_alphabet):
        if character not in vocabulary:
            vocabulary[character] = len(vocabulary)
            overflow_characters += 1

    diagnostics: dict[str, int] = {
        "nominal_vocab_size": vocab_size,
        "actual_vocab_size": len(vocabulary),
        "special_tokens": len(special_tokens),
        "subword_pieces": added_pieces,
        "overflow_characters": overflow_characters,
    }
    logger.info(
        "Vocabulary built: %d actual tokens (%d subwords + %d overflow chars)",
        len(vocabulary),
        added_pieces,
        overflow_characters,
    )
    return vocabulary, diagnostics


# Re-export for backward compatibility
from collections.abc import Iterable
