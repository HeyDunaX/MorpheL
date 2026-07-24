"""MorpheL configuration dataclasses.

All algorithm parameters are collected in :class:`MorpheLConfig`. The class is
frozen so that accidental mutation does not create non-reproducible states.

Design notes
------------
- ``vowels`` is language-dependent. It must be the **complete** orthographic
  vowel inventory of the target language, including uppercase variants when the
  corpus is not lowercased, and vowels with diacritics that occur in the
  writing system. Do not copy the Turkish inventory to another language.
- ``use_vowel_boundary_filter=False`` disables the vowel-transition heuristic.
  Disabling the filter is a method change that must be reported explicitly.
- ``temperature <= 0.0`` triggers deterministic argmax (T=0 mode) used for
  vocabulary induction and evaluation. This is not an approximation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from morphel.constants import (
    DEFAULT_MI_THRESHOLD,
    DEFAULT_MIN_FREQUENCY,
    DEFAULT_MODEL_MAX_LENGTH,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_VOCAB_SIZE,
    NUM_SPECIAL_TOKENS,
    SPECIAL_TOKENS,
)


@dataclass(frozen=True)
class MorpheLConfig:
    """Configuration for a MorpheL tokenizer.

    Args:
        language: Language identifier stored in ``morphel_config.json``.
            Must be a non-empty string, e.g. ``"tr"`` or ``"ru"``.
        vowels: Complete orthographic vowel inventory for the target language.
            Required when ``use_vowel_boundary_filter=True``.
        vocab_size: Nominal vocabulary size including special tokens.
            Actual vocabulary may be larger due to character-coverage overflow.
        min_frequency: Minimum corpus frequency for a subword piece to enter
            the nominal vocabulary. Must be at least 1.
        top_k: Number of top-MI candidate boundaries to consider per word.
            Must be at least 1.
        temperature: Gumbel sampling temperature. Values <= 0.0 produce
            deterministic argmax (T=0 mode).
        mi_threshold: Minimum MI score for a boundary to be retained.
            Boundaries with MI <= mi_threshold are discarded.
        model_max_length: Maximum sequence length stored in the tokenizer
            config; used by downstream HF models.
        seed: Random seed for NumPy. Ensures reproducibility of Gumbel
            sampling when temperature > 0.
        use_vowel_boundary_filter: If True, apply the vowel-transition
            heuristic to filter candidate boundaries. Set to False only when
            linguistically justified; report as an ablation.

    Raises:
        ValueError: On any constraint violation; see :meth:`validate`.
    """

    language: str
    vowels: str
    vocab_size: int = DEFAULT_VOCAB_SIZE
    min_frequency: int = DEFAULT_MIN_FREQUENCY
    top_k: int = DEFAULT_TOP_K
    temperature: float = DEFAULT_TEMPERATURE
    mi_threshold: float = DEFAULT_MI_THRESHOLD
    model_max_length: int = DEFAULT_MODEL_MAX_LENGTH
    seed: int = DEFAULT_SEED
    use_vowel_boundary_filter: bool = True

    def validate(self) -> None:
        """Validate all configuration constraints before expensive operations.

        Raises:
            ValueError: When any constraint is violated.
        """
        if not self.language.strip():
            raise ValueError(
                "language must be a non-empty language identifier, e.g. 'tr' or 'ru'."
            )
        if self.use_vowel_boundary_filter and not self.vowels:
            raise ValueError(
                "vowels must contain the target language's complete vowel inventory "
                "when use_vowel_boundary_filter=True. Include uppercase and diacritic "
                "variants that appear in the corpus. Do not copy the Turkish inventory "
                "to another language. To disable the filter, set "
                "use_vowel_boundary_filter=False and report that ablation."
            )
        if self.vocab_size <= NUM_SPECIAL_TOKENS:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must exceed the number of special "
                f"tokens ({NUM_SPECIAL_TOKENS})."
            )
        if self.min_frequency < 1:
            raise ValueError(
                f"min_frequency ({self.min_frequency}) must be at least 1."
            )
        if self.top_k < 1:
            raise ValueError(f"top_k ({self.top_k}) must be at least 1.")
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature ({self.temperature}) must be non-negative. "
                "Use 0.0 for deterministic argmax."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary suitable for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MorpheLConfig":
        """Construct from a plain dictionary (e.g., parsed from YAML or JSON).

        Unknown keys are silently ignored to allow forward compatibility.

        Args:
            data: Dictionary with configuration keys.

        Returns:
            A :class:`MorpheLConfig` instance.
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class TokenizerMetrics:
    """Intrinsic evaluation metrics for a MorpheL tokenizer.

    All rates and lengths are computed over the evaluation corpus. Words are
    defined by the tokenizer function used during induction.

    Args:
        fertility: Average tokens per whitespace word.
        tokens_per_char: Average tokens per character.
        avg_seq_len: Average encoded sequence length including BOS/EOS tokens.
        vocab_coverage: Fraction of word types with no OOV pieces.
        oov_rate: Fraction of word types with at least one OOV piece.
        fallback_event_rate: Fallback events per whitespace word.
        char_shatter_rate: Character-shatter events per whitespace word.
        sentences: Number of evaluation sentences.
        whitespace_words: Total whitespace words in evaluation corpus.
        word_types: Number of unique word types in evaluation corpus.
    """

    fertility: float
    tokens_per_char: float
    avg_seq_len: float
    vocab_coverage: float
    oov_rate: float
    fallback_event_rate: float
    char_shatter_rate: float
    sentences: int
    whitespace_words: int
    word_types: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)


@dataclass
class FallbackStats:
    """Counters for recursive fallback events during encoding.

    Args:
        fallback_events: Number of times a piece was not in the vocabulary and
            required recursive fallback decomposition.
        char_shatters: Number of times fallback could not find a vocabulary
            substring and fell back to individual characters.
    """

    fallback_events: int = 0
    char_shatters: int = 0
