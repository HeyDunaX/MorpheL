"""Unit tests for segmentation."""

from morphel.segmentation import segment_word


def test_segment_word_no_candidates() -> None:
    mi_index = {"apple": []}
    assert segment_word("apple", mi_index, temperature=0.0) == ["apple"]


def test_segment_word_deterministic_argmax() -> None:
    # "banana" has candidates at pos 2 and 4.
    mi_index = {
        "banana": [(2, 5.0), (4, 3.0)] # Top MI is pos 2.
    }
    # At T=0, it will pick the configuration with the highest gamma.
    # Gamma for zero cuts: 0.0
    # Gamma for one cut: 5.0 (the highest MI)
    # Gamma for two cuts: (5.0 + 3.0) / 2 = 4.0
    # Argmax is 1 cut, which corresponds to the first candidate (pos 2).
    # So the pieces should be "ba", "nana".
    pieces = segment_word("banana", mi_index, temperature=0.0)
    assert pieces == ["ba", "nana"]


def test_segment_word_special_tokens() -> None:
    assert segment_word("<s>", {}, temperature=0.0) == ["<s>"]
