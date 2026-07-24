"""Integration test for XLM-R pair template."""

from morphel.serialization import build_fast_tokenizer


def test_pair_template() -> None:
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4, "A": 5, "B": 6}
    fast_tokenizer = build_fast_tokenizer(vocab, 512)
    
    # Test single
    enc_single = fast_tokenizer.encode("A", add_special_tokens=True)
    # <s> A </s> -> [0, 5, 2]
    assert enc_single == [0, 5, 2]
    
    # Test pair
    enc_pair = fast_tokenizer.encode("A", "B", add_special_tokens=True)
    # <s> A </s></s> B </s> -> [0, 5, 2, 2, 6, 2]
    assert enc_pair == [0, 5, 2, 2, 6, 2]
