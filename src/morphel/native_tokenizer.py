"""Native MorpheL tokenizer — encoding and training orchestration.

This module implements the complete encoding pipeline and the top-level
training orchestrator. It is the primary entry point for library users.

Important limitation
---------------------
The Hugging Face WordLevel wrapper (``PreTrainedTokenizerFast``) saved by
:func:`morphel.serialization.save_native_artifacts` does NOT perform native
MorpheL segmentation on raw text. It expects pre-segmented input.

To encode raw text correctly, always use the native path::

    from morphel import load_native_tokenizer, encode_sentence_as_tokens

    artifacts = load_native_tokenizer("path/to/tokenizer/dir")
    tokens = encode_sentence_as_tokens(
        "Raw sentence text.",
        vocabulary=artifacts["vocabulary"],
        mi_index=artifacts["mi_index"],
        segmentation_cache=artifacts["segmentation_cache"],
        top_k=artifacts["config"].top_k,
    )

Do not use ``AutoTokenizer`` or the WordLevel model directly on raw text;
this silently replaces MorpheL segmentation with whitespace splitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from morphel.coalescing import coalesce_pieces
from morphel.config import FallbackStats, MorpheLConfig, TokenizerMetrics
from morphel.datasets import load_text_from_huggingface
from morphel.fallback import recursive_fallback
from morphel.logging_utils import get_logger
from morphel.metrics import compute_metrics
from morphel.mi import build_global_mi_index
from morphel.preprocessing import collect_word_frequencies, simple_tokenize
from morphel.sampling import set_seed
from morphel.segmentation import build_segmentation_cache, segment_word
from morphel.serialization import save_native_artifacts
from morphel.typing import MIIndex, SegmentationCache
from morphel.vocabulary import build_character_alphabet, build_vocabulary, count_contiguous_spans

logger = get_logger(__name__)


def encode_word(
    word: str,
    vocabulary: Mapping[str, int],
    mi_index: MIIndex,
    segmentation_cache: SegmentationCache,
    top_k: int,
    stats: Optional[FallbackStats] = None,
) -> list[str]:
    """Encode one orthographic word through cached segmentation, fallback, and coalescing.

    The encoding pipeline is:
    1. Look up the word in the deterministic segmentation cache.
       If absent, compute T=0 segmentation on the fly.
    2. Apply recursive longest-match fallback to each piece.
    3. Apply DP coalescing to minimize token count.

    Args:
        word: The orthographic word string to encode.
        vocabulary: Vocabulary mapping from token string -> ID.
        mi_index: Global MI index for on-the-fly segmentation of unseen words.
        segmentation_cache: Deterministic T=0 segmentation cache.
        top_k: Maximum MI candidates for on-the-fly segmentation.
        stats: Optional fallback statistics accumulator.

    Returns:
        List of token strings from the vocabulary.

    Determinism:
        Fully deterministic (uses T=0 segmentation from cache or on-the-fly).
    """
    raw_pieces = segmentation_cache.get(
        word,
        segment_word(word, mi_index, top_k=top_k, temperature=0.0),
    )
    expanded: list[str] = []
    for piece in raw_pieces:
        expanded.extend(recursive_fallback(piece, vocabulary, stats=stats))
    return coalesce_pieces(expanded, vocabulary)


def encode_sentence_as_tokens(
    sentence: str,
    vocabulary: Mapping[str, int],
    mi_index: MIIndex,
    segmentation_cache: SegmentationCache,
    top_k: int,
    tokenizer_fn: Callable[[str], list[str]] = simple_tokenize,
    stats: Optional[FallbackStats] = None,
) -> list[str]:
    """Encode raw text into native MorpheL token strings.

    This is the canonical raw-text encoding path. It applies the full
    MorpheL pipeline: word tokenization -> cached segmentation -> fallback
    -> coalescing.

    Args:
        sentence: Raw input text string.
        vocabulary: Vocabulary mapping from token string -> ID.
        mi_index: Global MI index.
        segmentation_cache: Deterministic T=0 segmentation cache.
        top_k: Maximum MI candidates for on-the-fly segmentation.
        tokenizer_fn: Function to split text into orthographic words. Must be
            the same function used during tokenizer induction.
        stats: Optional fallback statistics accumulator.

    Returns:
        List of token strings (not including BOS/EOS special tokens).

    Language assumption:
        The tokenizer function determines how raw text is split into words.
        Use a language-appropriate function and keep it consistent across all
        pipeline stages.
    """
    output: list[str] = []
    for word in tokenizer_fn(sentence):
        output.extend(
            encode_word(word, vocabulary, mi_index, segmentation_cache, top_k, stats=stats)
        )
    return output


def train_morphel(
    corpus_sentences: Sequence[str],
    config: MorpheLConfig,
    output_dir: Path,
    evaluation_sentences: Optional[Sequence[str]] = None,
    tokenizer_fn: Callable[[str], list[str]] = simple_tokenize,
) -> tuple[dict[str, int], MIIndex, SegmentationCache, Optional[TokenizerMetrics]]:
    """Train and save a complete native MorpheL tokenizer.

    Orchestrates the full pipeline:
    1. Validate configuration.
    2. Seed NumPy for reproducible Gumbel sampling.
    3. Collect word frequencies from the corpus.
    4. Build the global MI index.
    5. Build the deterministic T=0 segmentation cache.
    6. Count contiguous spans for vocabulary induction.
    7. Build the vocabulary (nominal + character overflow).
    8. Optionally compute intrinsic metrics on evaluation sentences.
    9. Save all native artifacts and the Hugging Face wrapper.

    Args:
        corpus_sentences: List of raw text strings for tokenizer induction.
        config: :class:`~morphel.config.MorpheLConfig` with all parameters.
        output_dir: Directory where native artifacts will be saved.
        evaluation_sentences: Optional list of sentences for intrinsic metric
            computation. If provided, metrics are saved to
            ``morphel_metrics_report.json``.
        tokenizer_fn: Word tokenization function. Must be consistent across all
            pipeline stages.

    Returns:
        Tuple of ``(vocabulary, mi_index, segmentation_cache, metrics)``.
        ``metrics`` is ``None`` if ``evaluation_sentences`` is not provided.

    Raises:
        ValueError: If ``config.validate()`` fails.
    """
    config.validate()
    set_seed(config.seed)
    vowels = set(config.vowels)

    logger.info("Collecting word frequencies from %d sentences…", len(corpus_sentences))
    word_frequency = collect_word_frequencies(corpus_sentences, tokenizer_fn)

    logger.info("Building global MI index…")
    mi_index = build_global_mi_index(
        word_frequency,
        vowels,
        mi_threshold=config.mi_threshold,
        use_vowel_boundary_filter=config.use_vowel_boundary_filter,
    )

    logger.info("Building T=0 segmentation cache…")
    segmentation_cache = build_segmentation_cache(
        word_frequency.keys(),
        mi_index,
        top_k=config.top_k,
    )

    logger.info("Counting contiguous spans…")
    subword_frequency = count_contiguous_spans(word_frequency, segmentation_cache)

    logger.info("Building vocabulary (vocab_size=%d)…", config.vocab_size)
    character_alphabet = build_character_alphabet(word_frequency.keys())
    vocabulary, vocabulary_diagnostics = build_vocabulary(
        subword_frequency,
        character_alphabet,
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
    )

    for key, value in vocabulary_diagnostics.items():
        logger.info("  %s: %d", key, value)

    metrics: Optional[TokenizerMetrics] = None
    if evaluation_sentences is not None:
        logger.info("Computing intrinsic metrics on %d sentences…", len(evaluation_sentences))
        metrics = compute_metrics(
            list(evaluation_sentences),
            vocabulary,
            mi_index,
            segmentation_cache,
            top_k=config.top_k,
            tokenizer_fn=tokenizer_fn,
        )
        logger.info("Tokenizer metrics:")
        for key, value in metrics.to_dict().items():
            logger.info("  %s: %s", key, value)

    logger.info("Saving native artifacts to %s…", output_dir)
    save_native_artifacts(output_dir, config, vocabulary, mi_index, segmentation_cache, metrics)

    return vocabulary, mi_index, segmentation_cache, metrics
