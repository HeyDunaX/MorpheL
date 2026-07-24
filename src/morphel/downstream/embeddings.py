"""Embedding initialization strategies (WECHSEL).

The paper relies on WECHSEL or similar mapping procedures to initialize the
word embeddings of the new tokenizer using the pretrained model's source
embeddings.
"""

from __future__ import annotations

from typing import Any, Mapping


class EmbeddingInitializer:
    """Interface for word embedding initialization.

    The official WECHSEL implementation is not included in the handoff bundle
    and will be supplied by the project owner.
    """

    @staticmethod
    def initialize(
        source_model: Any,
        target_vocabulary: Mapping[str, int],
        source_language: str,
        target_language: str,
        seed: int,
    ) -> Any:  # Returns numpy.ndarray
        """Initialize target embeddings from a source model.

        Downloads of multi-gigabyte fastText files must happen here, not at
        import time, and should support a cache directory.

        Args:
            source_model: The pretrained HF model with source embeddings.
            target_vocabulary: The native MorpheL vocabulary mapping.
            source_language: Language code of the source model (e.g., 'en').
            target_language: Language code of the target (e.g., 'tr').
            seed: Random seed for resolving ties or random fallbacks.

        Returns:
            A NumPy array of shape (len(target_vocabulary), embedding_dim).

        Raises:
            NotImplementedError: Until the project owner supplies the code.
        """
        raise NotImplementedError("Stub: pending project owner implementation of WECHSEL.")
