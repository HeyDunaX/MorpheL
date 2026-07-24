"""Unit tests for DP coalescing."""

from morphel.coalescing import coalesce_pieces


def test_coalesce_pieces_no_merge_needed() -> None:
    pieces = ["a", "b", "c"]
    vocab = {"a": 1, "b": 2, "c": 3}
    assert coalesce_pieces(pieces, vocab) == ["a", "b", "c"]


def test_coalesce_pieces_merge_all() -> None:
    pieces = ["a", "p", "p", "l", "e"]
    vocab = {"a": 1, "p": 2, "l": 3, "e": 4, "app": 5, "le": 6, "apple": 7}
    # The optimal grouping is just "apple" (1 token)
    assert coalesce_pieces(pieces, vocab) == ["apple"]


def test_coalesce_pieces_partial_merge() -> None:
    pieces = ["a", "p", "p", "l", "e"]
    vocab = {"a": 1, "p": 2, "l": 3, "e": 4, "app": 5, "le": 6}
    # The optimal grouping is "app", "le" (2 tokens)
    assert coalesce_pieces(pieces, vocab) == ["app", "le"]


def test_coalesce_pieces_invalid_merge() -> None:
    pieces = ["x", "y"]
    vocab = {} # Invalid vocab, neither piece is present
    # DP will fail, should return original list safely
    assert coalesce_pieces(pieces, vocab) == ["x", "y"]
