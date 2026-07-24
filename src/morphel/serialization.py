"""Serialization and loading of native MorpheL artifacts.

Native artifacts
----------------
The following files constitute a complete native MorpheL tokenizer:

- ``morphel_vocab.json``    — token-to-ID mapping (JSON)
- ``mi_index.pkl``          — global MI index (pickle)
- ``segmentation_cache.pkl`` — deterministic T=0 segmentations (pickle)
- ``morphel_config.json``   — algorithm configuration (JSON)
- ``morphel_metrics_report.json`` — intrinsic metrics (JSON; optional)

The Hugging Face wrapper (``tokenizer.json``, ``tokenizer_config.json``, etc.)
is saved alongside these artifacts but does NOT reproduce native MorpheL
segmentation on raw text. Always load native artifacts for correct encoding.

Limitation
----------
``AutoTokenizer.from_pretrained(dir)`` loads the WordLevel wrapper, which
splits raw text on whitespace and looks up each whitespace token as a whole
word. This silently replaces the MorpheL segmentation pipeline and will produce
incorrect encodings for most input. Load native artifacts via
:func:`load_native_tokenizer` instead.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from morphel.config import MorpheLConfig, TokenizerMetrics
from morphel.constants import (
    ARTIFACT_CACHE,
    ARTIFACT_CONFIG,
    ARTIFACT_MI_INDEX,
    ARTIFACT_METRICS,
    ARTIFACT_VOCAB,
    PAIR_TEMPLATE_PAIR,
    PAIR_TEMPLATE_SINGLE,
    SPECIAL_TOKENS,
)
from morphel.logging_utils import get_logger
from morphel.typing import MIIndex, SegmentationCache

logger = get_logger(__name__)


def build_fast_tokenizer(
    vocabulary: dict[str, int],
    model_max_length: int,
) -> Any:  # Returns PreTrainedTokenizerFast
    """Build the WordLevel Hugging Face wrapper around native MorpheL pieces.

    The wrapper handles BOS/EOS insertion and padding but does NOT perform
    MorpheL segmentation on raw text. Downstream users must load native
    artifacts and run the native segmentation path.

    Args:
        vocabulary: Token-to-ID mapping from vocabulary induction.
        model_max_length: Maximum sequence length for the HF config.

    Returns:
        A :class:`transformers.PreTrainedTokenizerFast` instance.
    """
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(vocab=vocabulary, unk_token="<unk>"))
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single=PAIR_TEMPLATE_SINGLE,
        pair=PAIR_TEMPLATE_PAIR,
        special_tokens=[
            ("<s>", vocabulary["<s>"]),
            ("</s>", vocabulary["</s>"]),
        ],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask>",
        model_max_length=model_max_length,
    )


def save_native_artifacts(
    output_dir: Path,
    config: MorpheLConfig,
    vocabulary: dict[str, int],
    mi_index: MIIndex,
    segmentation_cache: SegmentationCache,
    metrics: Optional[TokenizerMetrics] = None,
) -> None:
    """Save all native MorpheL artifacts and the Hugging Face wrapper.

    Creates the output directory if it does not exist.

    Args:
        output_dir: Target directory for all artifact files.
        config: :class:`~morphel.config.MorpheLConfig` used during training.
        vocabulary: Token-to-ID mapping.
        mi_index: Global MI index.
        segmentation_cache: Deterministic T=0 segmentation cache.
        metrics: Optional :class:`~morphel.config.TokenizerMetrics` to save.
            Saved as ``morphel_metrics_report.json`` when provided.

    Files written:
        - ``morphel_vocab.json``
        - ``mi_index.pkl``
        - ``segmentation_cache.pkl``
        - ``morphel_config.json``
        - ``morphel_metrics_report.json`` (if metrics is not None)
        - HF wrapper files (tokenizer.json, tokenizer_config.json, …)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / ARTIFACT_VOCAB).open("w", encoding="utf-8") as fh:
        json.dump(vocabulary, fh, ensure_ascii=False, indent=2)

    with (output_dir / ARTIFACT_MI_INDEX).open("wb") as fh:
        pickle.dump(mi_index, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with (output_dir / ARTIFACT_CACHE).open("wb") as fh:
        pickle.dump(segmentation_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)

    config_payload = asdict(config)
    config_payload.update(
        {
            "special_tokens": list(SPECIAL_TOKENS),
            "actual_vocab_size": len(vocabulary),
            "segmentation_cache_temperature": 0.0,
            "native_segmentation_required": True,
        }
    )
    with (output_dir / ARTIFACT_CONFIG).open("w", encoding="utf-8") as fh:
        json.dump(config_payload, fh, ensure_ascii=False, indent=2)

    if metrics is not None:
        with (output_dir / ARTIFACT_METRICS).open("w", encoding="utf-8") as fh:
            json.dump(asdict(metrics), fh, ensure_ascii=False, indent=2)

    fast_tokenizer = build_fast_tokenizer(vocabulary, config.model_max_length)
    fast_tokenizer.save_pretrained(str(output_dir))
    logger.info("Saved native artifacts and HF wrapper to %s", output_dir)


def load_native_tokenizer(
    tokenizer_dir: Path | str,
) -> dict[str, Any]:
    """Load all native MorpheL artifacts from a saved tokenizer directory.

    This is the correct way to load a MorpheL tokenizer for encoding raw text.
    Do not use ``AutoTokenizer.from_pretrained`` for this purpose.

    Args:
        tokenizer_dir: Path to the saved tokenizer directory containing all
            required native artifacts.

    Returns:
        Dictionary with keys:
        - ``"vocabulary"`` — token-to-ID mapping
        - ``"mi_index"`` — global MI index
        - ``"segmentation_cache"`` — deterministic T=0 segmentation cache
        - ``"config"`` — :class:`~morphel.config.MorpheLConfig` instance
        - ``"metrics"`` — :class:`~morphel.config.TokenizerMetrics` or None

    Raises:
        FileNotFoundError: If any required native artifact is missing.
        ValueError: If the loaded config fails validation.
    """
    tokenizer_dir = Path(tokenizer_dir)

    for artifact in (ARTIFACT_VOCAB, ARTIFACT_MI_INDEX, ARTIFACT_CACHE, ARTIFACT_CONFIG):
        artifact_path = tokenizer_dir / artifact
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Required native artifact not found: {artifact_path}. "
                "The Hugging Face wrapper alone cannot reproduce native MorpheL "
                "segmentation. Ensure the complete artifact directory is present."
            )

    with (tokenizer_dir / ARTIFACT_VOCAB).open("r", encoding="utf-8") as fh:
        vocabulary: dict[str, int] = json.load(fh)

    with (tokenizer_dir / ARTIFACT_MI_INDEX).open("rb") as fh:
        mi_index: MIIndex = pickle.load(fh)  # noqa: S301

    with (tokenizer_dir / ARTIFACT_CACHE).open("rb") as fh:
        segmentation_cache: SegmentationCache = pickle.load(fh)  # noqa: S301

    with (tokenizer_dir / ARTIFACT_CONFIG).open("r", encoding="utf-8") as fh:
        config_data: dict[str, Any] = json.load(fh)
    config = MorpheLConfig.from_dict(config_data)

    metrics: Optional[TokenizerMetrics] = None
    metrics_path = tokenizer_dir / ARTIFACT_METRICS
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fh:
            metrics_data = json.load(fh)
        metrics = TokenizerMetrics(**metrics_data)

    logger.info(
        "Loaded native MorpheL tokenizer: vocab=%d, cache=%d words",
        len(vocabulary),
        len(segmentation_cache),
    )
    return {
        "vocabulary": vocabulary,
        "mi_index": mi_index,
        "segmentation_cache": segmentation_cache,
        "config": config,
        "metrics": metrics,
    }
