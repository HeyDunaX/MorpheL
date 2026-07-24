"""Integration test verifying end-to-end tokenizer training on tiny corpus."""

from morphel.config import MorpheLConfig
from morphel.native_tokenizer import train_morphel


def test_train_tiny_tokenizer(tiny_corpus, tmp_path) -> None:
    config = MorpheLConfig(
        language="en",
        vowels="aeiouAEIOU",
        vocab_size=100,
        min_frequency=1,
        top_k=2,
    )
    vocab, mi, cache, metrics = train_morphel(
        corpus_sentences=tiny_corpus,
        config=config,
        output_dir=tmp_path,
        evaluation_sentences=tiny_corpus,
    )
    
    assert len(vocab) > 50
    assert metrics is not None
    assert metrics.sentences == len(tiny_corpus)
    # <s>, <pad>, </s>, <unk>, <mask> should be exactly indices 0-4
    assert vocab["<s>"] == 0
    assert vocab["<pad>"] == 1
    assert vocab["</s>"] == 2
    assert vocab["<unk>"] == 3
    assert vocab["<mask>"] == 4
