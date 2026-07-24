"""Pre-flight argument validation for MorpheL CLI commands.

All expensive operations (dataset downloads, model loading) are preceded by
validation to catch configuration errors early.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from morphel.logging_utils import get_logger

logger = get_logger(__name__)


class ValidationError(ValueError):
    """Raised when a CLI argument or configuration value fails pre-flight checks."""


def require_nonempty(value: str, name: str) -> None:
    """Raise ValidationError if ``value`` is empty or whitespace-only."""
    if not value or not value.strip():
        raise ValidationError(f"--{name} must not be empty.")


def require_positive_int(value: int, name: str) -> None:
    """Raise ValidationError if ``value`` is less than 1."""
    if value < 1:
        raise ValidationError(f"--{name} must be at least 1 (got {value}).")


def require_nonnegative_float(value: float, name: str) -> None:
    """Raise ValidationError if ``value`` is negative."""
    if value < 0.0:
        raise ValidationError(f"--{name} must be non-negative (got {value}).")


def require_output_dir_fresh(output_dir: Path, overwrite: bool) -> None:
    """Raise ValidationError if the output directory exists and overwrite is False.

    Args:
        output_dir: Proposed output directory.
        overwrite: Whether to allow overwriting existing outputs.

    Raises:
        ValidationError: If the directory exists and overwrite is False.
    """
    if output_dir.exists() and not overwrite:
        raise ValidationError(
            f"Output directory already exists: {output_dir}. "
            "Pass --overwrite to replace it."
        )


def require_vowels_when_filter_enabled(
    vowels: str,
    use_vowel_boundary_filter: bool,
) -> None:
    """Raise ValidationError if the vowel filter is enabled but no vowels supplied."""
    if use_vowel_boundary_filter and not vowels:
        raise ValidationError(
            "--vowels must be supplied when the vowel boundary filter is enabled. "
            "Provide the complete vowel inventory of the target language. "
            "To disable the filter, pass --disable-vowel-boundary-filter and "
            "report that change as an ablation."
        )


def require_vocab_size_exceeds_special_tokens(
    vocab_size: int,
    num_special_tokens: int,
) -> None:
    """Raise ValidationError if vocab_size is too small for special tokens."""
    if vocab_size <= num_special_tokens:
        raise ValidationError(
            f"--vocab-size ({vocab_size}) must exceed the number of special "
            f"tokens ({num_special_tokens})."
        )


def validate_train_tokenizer_args(
    language: str,
    vowels: str,
    use_vowel_boundary_filter: bool,
    vocab_size: int,
    min_frequency: int,
    top_k: int,
    temperature: float,
    output_dir: Path,
    overwrite: bool,
    splits: Sequence[str],
    num_special_tokens: int,
) -> None:
    """Run all pre-flight checks for the train-tokenizer command.

    Raises:
        ValidationError: On the first constraint that fails.
    """
    require_nonempty(language, "language")
    require_vowels_when_filter_enabled(vowels, use_vowel_boundary_filter)
    require_vocab_size_exceeds_special_tokens(vocab_size, num_special_tokens)
    require_positive_int(min_frequency, "min-frequency")
    require_positive_int(top_k, "top-k")
    require_nonnegative_float(temperature, "temperature")
    require_output_dir_fresh(output_dir, overwrite)
    if not splits:
        raise ValidationError("--splits must contain at least one split name.")
