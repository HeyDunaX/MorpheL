"""Integration test comparing native encoding vs WordLevel wrapper limitation."""

from morphel.config import MorpheLConfig
from morphel.native_tokenizer import encode_sentence_as_tokens, train_morphel
from morphel.serialization import build_fast_tokenizer, load_native_tokenizer


def test_native_wrapper_limitation(tmp_path) -> None:
    """Verify that the HF wrapper does NOT perform MorpheL segmentation on raw text.
    
    This documents the limitation explicitly. The HF wrapper splits on whitespace
    and tries to look up the whole word. The native path segments it.
    """
    corpus = ["applebanana"]
    
    config = MorpheLConfig(
        language="en",
        vowels="aeiou",
        vocab_size=50,
        min_frequency=1,
        top_k=2,
        use_vowel_boundary_filter=False, # to allow arbitrary splits
    )
    
    vocab, mi, cache, metrics = train_morphel(
        corpus_sentences=corpus,
        config=config,
        output_dir=tmp_path,
    )
    
    # Load native
    artifacts = load_native_tokenizer(tmp_path)
    # The native path will encode 'applebanana' using the generated vocabulary.
    # It might split it if 'applebanana' itself was not retained as a whole token.
    # But wait, 'applebanana' is in the corpus, so it's in the vocab if vocab_size permits.
    # Let's use a word not in the corpus: 'apple'
    
    native_tokens = encode_sentence_as_tokens(
        "apple",
        vocabulary=artifacts["vocabulary"],
        mi_index=artifacts["mi_index"],
        segmentation_cache=artifacts["segmentation_cache"],
        top_k=artifacts["config"].top_k,
    )
    
    # The HF wrapper
    fast_tokenizer = build_fast_tokenizer(artifacts["vocabulary"], 512)
    hf_tokens = fast_tokenizer.tokenize("apple")
    
    # HF wrapper will output ['<unk>'] if 'apple' wasn't in vocab,
    # Native path will shatter it to characters ['a', 'p', 'p', 'l', 'e'].
    # Just assert they are different when given unseen words.
    if "apple" not in vocab:
        assert hf_tokens == ["<unk>"]
        assert native_tokens != ["<unk>"]
