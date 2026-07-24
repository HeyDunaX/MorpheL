"""Unit tests for vocabulary building."""

from collections import Counter

from morphel.vocabulary import build_character_alphabet, build_vocabulary, count_contiguous_spans


def test_build_character_alphabet() -> None:
    words = ["apple", "banana"]
    alphabet = build_character_alphabet(words)
    assert alphabet == set("apleban")


def test_count_contiguous_spans() -> None:
    word_freq = {"banana": 10}
    # Pretend cache says "banana" -> "ba", "na", "na"
    cache = {"banana": ["ba", "na", "na"]}
    
    spans = count_contiguous_spans(word_freq, cache)
    
    # Expected spans:
    # Length 1: "ba" (10), "na" (10 + 10 = 20)
    # Length 2: "bana" (10), "nana" (10)
    # Length 3: "banana" (10)
    assert spans["ba"] == 10
    assert spans["na"] == 20
    assert spans["bana"] == 10
    assert spans["nana"] == 10
    assert spans["banana"] == 10


def test_build_vocabulary() -> None:
    spans = Counter({"a": 100, "b": 50, "c": 20})
    alphabet = set("abcd") # 'd' is missing from spans
    
    vocab, diag = build_vocabulary(
        subword_frequency=spans,
        character_alphabet=alphabet,
        vocab_size=10,
        min_frequency=1,
        special_tokens=["<s>", "</s>"],
    )
    
    assert vocab["<s>"] == 0
    assert vocab["</s>"] == 1
    assert vocab["a"] == 2
    assert vocab["b"] == 3
    assert vocab["c"] == 4
    # 'd' was appended as character coverage overflow
    assert vocab["d"] == 5
    
    assert diag["overflow_characters"] == 1
    assert diag["actual_vocab_size"] == 6
