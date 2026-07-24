"""Intrinsic evaluation metrics for MorpheL tokenizers.

Metrics are computed by encoding the evaluation corpus through the full native
MorpheL pipeline and counting tokens, fallback events, and OOV word types.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from tqdm.auto import tqdm

from morphel.coalescing import coalesce_pieces
from morphel.config import FallbackStats, TokenizerMetrics
from morphel.fallback import recursive_fallback
from morphel.logging_utils import get_logger
# Removed circular import of encode_word
from morphel.preprocessing import simple_tokenize
from morphel.typing import MIIndex, SegmentationCache

logger = get_logger(__name__)


def compute_metrics(
    sentences: Sequence[str],
    vocabulary: Mapping[str, int],
    mi_index: MIIndex,
    segmentation_cache: SegmentationCache,
    top_k: int,
    tokenizer_fn: Callable[[str], list[str]] = simple_tokenize,
) -> TokenizerMetrics:
    """Compute intrinsic evaluation metrics for a MorpheL tokenizer.

    Metrics are computed over the provided evaluation sentences using the
    native MorpheL encoding path (cached segmentation + fallback + coalescing).

    Computed metrics:
    - ``fertility``: average tokens per whitespace word.
    - ``tokens_per_char``: average tokens per character.
    - ``avg_seq_len``: average encoded sequence length including BOS/EOS tokens.
    - ``vocab_coverage``: fraction of word types with no OOV pieces.
    - ``oov_rate``: fraction of word types with at least one OOV piece.
    - ``fallback_event_rate``: fallback events per whitespace word.
    - ``char_shatter_rate``: character-shatter events per whitespace word.

    Args:
        sentences: List of raw text strings to evaluate.
        vocabulary: Token-to-ID mapping.
        mi_index: Global MI index.
        segmentation_cache: Deterministic T=0 segmentation cache.
        top_k: Maximum MI candidates for on-the-fly segmentation.
        tokenizer_fn: Word tokenization function. Must be identical to the
            function used during tokenizer induction.

    Returns:
        A :class:`~morphel.config.TokenizerMetrics` instance.
    """
    total_words = 0
    total_characters = 0
    total_tokens = 0
    total_sequence_tokens = 0
    all_word_types: set[str] = set()
    oov_word_types: set[str] = set()
    stats = FallbackStats()

    for sentence in tqdm(sentences, desc="Evaluating tokenizer"):
        words = tokenizer_fn(sentence)
        if not words:
            continue

        from morphel.native_tokenizer import encode_word
        total_words += len(words)
        total_characters += sum(len(word) for word in words)
        all_word_types.update(words)

        sentence_token_count = 0
        for word in words:
            pieces = encode_word(
                word,
                vocabulary,
                mi_index,
                segmentation_cache,
                top_k,
                stats=stats,
            )
            sentence_token_count += len(pieces)
            if any(piece not in vocabulary for piece in pieces):
                oov_word_types.add(word)

        total_tokens += sentence_token_count
        total_sequence_tokens += sentence_token_count + 2  # +2 for BOS/EOS

    word_type_count = len(all_word_types)
    oov_rate = len(oov_word_types) / max(word_type_count, 1)

    return TokenizerMetrics(
        fertility=total_tokens / max(total_words, 1),
        tokens_per_char=total_tokens / max(total_characters, 1),
        avg_seq_len=total_sequence_tokens / max(len(sentences), 1),
        vocab_coverage=1.0 - oov_rate,
        oov_rate=oov_rate,
        fallback_event_rate=stats.fallback_events / max(total_words, 1),
        char_shatter_rate=stats.char_shatters / max(total_words, 1),
        sentences=len(sentences),
        whitespace_words=total_words,
        word_types=word_type_count,
    )
