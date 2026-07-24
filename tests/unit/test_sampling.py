"""Unit tests for sampling."""

import pytest

from morphel.sampling import gumbel_sample, set_seed


def test_gumbel_sample_deterministic_at_zero() -> None:
    logits = [1.0, 5.0, 2.0]
    # At temperature <= 0, should always be argmax (index 1)
    assert gumbel_sample(logits, temperature=0.0) == 1
    assert gumbel_sample(logits, temperature=-1.0) == 1


def test_gumbel_sample_stochastic() -> None:
    logits = [1.0, 5.0, 2.0]
    # Set seed so we get a reproducible but "stochastic" result
    set_seed(42)
    # The argmax without noise is 1. With noise, it could be something else,
    # but we just want to ensure it runs without crashing.
    result = gumbel_sample(logits, temperature=1.0)
    assert result in [0, 1, 2]


def test_gumbel_sample_empty_logits_raises() -> None:
    with pytest.raises(ValueError):
        gumbel_sample([], temperature=1.0)
