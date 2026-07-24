"""Unit tests for preprocessing."""

from morphel.preprocessing import collect_word_frequencies, simple_tokenize


def test_simple_tokenize_handles_punctuation() -> None:
    text = "Hello, world! This is a test."
    assert simple_tokenize(text) == ["Hello", "world", "This", "is", "a", "test"]


def test_simple_tokenize_preserves_internal_punctuation() -> None:
    text = "don't multi-word"
    assert simple_tokenize(text) == ["don't", "multi-word"]


def test_simple_tokenize_handles_sentences() -> None:
    text = "Sentence one. Sentence two? Sentence three!"
    assert simple_tokenize(text) == ["Sentence", "one", "Sentence", "two", "Sentence", "three"]


def test_collect_word_frequencies() -> None:
    sentences = ["apple banana", "apple cherry", "banana banana"]
    freq = collect_word_frequencies(sentences, simple_tokenize)
    assert freq["apple"] == 2
    assert freq["banana"] == 3
    assert freq["cherry"] == 1
    assert "cherry," not in freq
