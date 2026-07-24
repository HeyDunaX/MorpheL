"""Package-wide constants for MorpheL.

Do not import from this module in ways that trigger network calls or
heavy computation at import time.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Special tokens — XLM-R-compatible order
# ---------------------------------------------------------------------------
# The order is non-negotiable. Downstream encoders that replace XLM-R's
# word-embedding matrix expect IDs 0-4 to match these tokens exactly.
SPECIAL_TOKENS: Final[tuple[str, ...]] = (
    "<s>",     # BOS / CLS — ID 0
    "<pad>",   # Padding   — ID 1
    "</s>",    # EOS / SEP — ID 2
    "<unk>",   # Unknown   — ID 3
    "<mask>",  # Mask      — ID 4
)

NUM_SPECIAL_TOKENS: Final[int] = len(SPECIAL_TOKENS)

# ---------------------------------------------------------------------------
# Default algorithm hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_VOCAB_SIZE: Final[int] = 32_000
DEFAULT_MIN_FREQUENCY: Final[int] = 2
DEFAULT_TOP_K: Final[int] = 4
DEFAULT_TEMPERATURE: Final[float] = 1.0
DEFAULT_MI_THRESHOLD: Final[float] = 0.0
DEFAULT_MODEL_MAX_LENGTH: Final[int] = 512
DEFAULT_SEED: Final[int] = 42

# ---------------------------------------------------------------------------
# Native artifact file names
# ---------------------------------------------------------------------------
ARTIFACT_VOCAB: Final[str] = "morphel_vocab.json"
ARTIFACT_MI_INDEX: Final[str] = "mi_index.pkl"
ARTIFACT_CACHE: Final[str] = "segmentation_cache.pkl"
ARTIFACT_CONFIG: Final[str] = "morphel_config.json"
ARTIFACT_METRICS: Final[str] = "morphel_metrics_report.json"

REQUIRED_NATIVE_ARTIFACTS: Final[tuple[str, ...]] = (
    ARTIFACT_VOCAB,
    ARTIFACT_MI_INDEX,
    ARTIFACT_CACHE,
    ARTIFACT_CONFIG,
)

# ---------------------------------------------------------------------------
# Pair encoding template — matches XLM-R convention
# ---------------------------------------------------------------------------
PAIR_TEMPLATE_SINGLE: Final[str] = "<s> $A </s>"
PAIR_TEMPLATE_PAIR: Final[str] = "<s> $A </s> </s> $B </s>"
