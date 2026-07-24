"""Dataset loading utilities for MorpheL.

All dataset access goes through this module to ensure consistent handling of
text columns, missing values, and dataset revision pinning.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Mapping, Optional, Sequence

from morphel.logging_utils import get_logger

logger = get_logger(__name__)


def iter_dataset_text(
    dataset: Iterable[Mapping[str, object]],
    text_columns: Sequence[str],
) -> Iterator[str]:
    """Yield non-empty strings from the requested dataset columns.

    Args:
        dataset: Iterable of dataset rows (each row is a mapping).
        text_columns: Names of text columns to extract.

    Yields:
        Non-empty, stripped string values from each column.
    """
    for row in dataset:
        for column in text_columns:
            value = str(row.get(column) or "").strip()
            if value:
                yield value


def load_text_from_huggingface(
    dataset_name: str,
    subset: Optional[str],
    splits: Sequence[str],
    text_columns: Sequence[str],
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    num_workers: int = 1,
) -> list[str]:
    """Load and concatenate text columns from Hugging Face datasets.

    Args:
        dataset_name: Hugging Face dataset identifier (e.g. ``"facebook/xnli"``).
        subset: Dataset subset or language code (e.g. ``"tr"``).
            Pass ``None`` for datasets without subsets.
        splits: List of split names to load and concatenate
            (e.g. ``["train", "validation", "test"]``).
        text_columns: Column names containing the text to extract.
        revision: Optional dataset revision (commit hash or tag) for pinning.
        cache_dir: Optional local directory for the Hugging Face datasets cache.
        num_workers: Number of parallel workers for dataset preprocessing.

    Returns:
        List of all non-empty text strings from the requested columns and splits.

    Raises:
        ValueError: If ``splits`` is empty.
    """
    from datasets import concatenate_datasets, load_dataset

    if not splits:
        raise ValueError("splits must be a non-empty list of split names.")

    logger.info(
        "Loading dataset %s (subset=%s, splits=%s, revision=%s)",
        dataset_name,
        subset,
        splits,
        revision,
    )

    loaded = []
    for split in splits:
        kwargs: dict[str, object] = {
            "split": split,
            "num_proc": num_workers,
        }
        if revision is not None:
            kwargs["revision"] = revision
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir

        if subset:
            ds = load_dataset(dataset_name, subset, **kwargs)
        else:
            ds = load_dataset(dataset_name, **kwargs)
        loaded.append(ds)

    combined = concatenate_datasets(loaded) if len(loaded) > 1 else loaded[0]
    sentences = list(iter_dataset_text(combined, text_columns))
    logger.info("Loaded %d text fields from %s", len(sentences), dataset_name)
    return sentences
