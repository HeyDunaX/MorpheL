"""Unit tests for recursive fallback."""

from morphel.config import FallbackStats
from morphel.fallback import recursive_fallback


def test_recursive_fallback_in_vocab() -> None:
    vocab = {"apple": 1}
    assert recursive_fallback("apple", vocab) == ["apple"]


def test_recursive_fallback_prefix_match() -> None:
    vocab = {"app": 1, "le": 2}
    # "apple" not in vocab.
    # Longest prefix is "app". Remainder is "le".
    assert recursive_fallback("apple", vocab) == ["app", "le"]


def test_recursive_fallback_suffix_match() -> None:
    vocab = {"pple": 1, "a": 2}
    # Longest prefix fails, tries suffix "pple". Head is "a".
    assert recursive_fallback("apple", vocab) == ["a", "pple"]


def test_recursive_fallback_shatter() -> None:
    vocab = {"<s>": 1} # Extremely tiny vocab
    stats = FallbackStats()
    # Should shatter to characters
    assert recursive_fallback("apple", vocab, stats=stats) == ["a", "p", "p", "l", "e"]
    assert stats.fallback_events > 0
    assert stats.char_shatters > 0
