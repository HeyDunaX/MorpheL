"""Unit tests for native tokenizer orchestration."""

import pytest

from morphel.config import MorpheLConfig
from morphel.native_tokenizer import encode_sentence_as_tokens, train_morphel
from morphel.serialization import load_native_tokenizer


def test_train_save_load_encode_roundtrip(tmp_path) -> None:
    """Integration test verifying full pipeline orchestration."""
    corpus = [
        "This is a test corpus.",
        "It contains some simple sentences.",
        "MorpheL should tokenize this gracefully.",
    ]

    config = MorpheLConfig(
        language="en",
        vowels="aeiouAEIOU",
        vocab_size=50,
        min_frequency=1,
        top_k=2,
    )

    # Train
    vocab, mi, cache, metrics = train_morphel(
        corpus_sentences=corpus,
        config=config,
        output_dir=tmp_path,
        evaluation_sentences=corpus,
    )

    assert len(vocab) > 0
    assert metrics is not None
    assert (tmp_path / "morphel_vocab.json").exists()

    # Load
    artifacts = load_native_tokenizer(tmp_path)
    assert artifacts["vocabulary"] == vocab
    assert artifacts["config"].language == "en"

    # Encode
    sentence = "This is a simple test."
    tokens = encode_sentence_as_tokens(
        sentence,
        vocabulary=artifacts["vocabulary"],
        mi_index=artifacts["mi_index"],
        segmentation_cache=artifacts["segmentation_cache"],
        top_k=artifacts["config"].top_k,
    )
    assert isinstance(tokens, list)
    assert len(tokens) > 0
