"""Text preprocessing for MorpheL tokenizer induction.

The default implementation assumes whitespace-delimited orthographic words,
which is appropriate for most languages with explicit word boundaries.

Language-specific requirements
-------------------------------
For languages **without** reliable whitespace word boundaries (Chinese,
Japanese, Thai, and similar writing systems), researchers MUST:

1. Provide a custom ``TokenizerFunction`` that implements the appropriate
   word or character-unit segmentation.
2. Register the function and pass it consistently to:
   - :func:`collect_word_frequencies` (induction),
   - :func:`morphel.native_tokenizer.encode_sentence_as_tokens` (encoding),
   - all downstream evaluation scripts.
3. Report the segmentation policy explicitly in the paper and repository.

The ``--tokenizer-function`` CLI argument accepts a fully-qualified import
path, e.g. ``mypackage.preprocessing:jieba_tokenize``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Iterable

from tqdm.auto import tqdm

from morphel.logging_utils import get_logger
from morphel.normalization import strip_boundary_punctuation

logger = get_logger(__name__)


def simple_tokenize(text: str) -> list[str]:
    """Tokenize text into whitespace-delimited orthographic words.

    Sentence-final punctuation (``. ! ? ; :``), when found inside a whitespace
    token, is treated as a separator. Boundary punctuation is removed using
    Unicode character categories.

    Language assumption: This function assumes that whitespace reliably
    separates orthographic words. For languages without such a convention
    (Chinese, Japanese, Thai, and similar writing systems), replace this
    function with an appropriate, declared word-segmentation procedure, and
    pass the same function to every downstream loader.

    Args:
        text: Raw input text string.

    Returns:
        List of cleaned orthographic word strings. Empty strings are excluded.

    Example:
        >>> simple_tokenize("Hello, world! This is a test.")
        ['Hello', 'world', 'This', 'is', 'a', 'test']
        >>> simple_tokenize("don't stop")
        ["don't", 'stop']
    """
    words: list[str] = []
    for raw_token in str(text).split():
        for part in re.split(r"(?<=[^\s])[.!?;:](?=[^\s])", raw_token):
            cleaned = strip_boundary_punctuation(part)
            if cleaned:
                words.append(cleaned)
    return words


def collect_word_frequencies(
    corpus_sentences: Iterable[str],
    tokenizer_fn: Callable[[str], list[str]] = simple_tokenize,
) -> Counter:
    """Collect corpus-token frequencies for orthographic word types.

    Frequencies are weighted by the number of times each word type appears
    across the entire corpus. These weighted counts are used in the global
    prefix/suffix MI calculation.

    Args:
        corpus_sentences: Iterable of raw text strings (one sentence each).
        tokenizer_fn: Function mapping a raw text string to a list of
            orthographic word strings. Defaults to :func:`simple_tokenize`.
            Replace with a language-appropriate function when needed.

    Returns:
        A :class:`collections.Counter` mapping word string -> corpus frequency.

    Determinism:
        This function is deterministic given the same inputs.
    """
    frequencies: Counter = Counter()
    for sentence in tqdm(corpus_sentences, desc="Collecting word frequencies"):
        frequencies.update(tokenizer_fn(sentence))
    total = sum(frequencies.values())
    logger.info(
        "Word frequencies collected: %d types, %d tokens", len(frequencies), total
    )
    return frequencies
