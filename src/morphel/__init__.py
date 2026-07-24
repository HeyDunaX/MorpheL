"""MorpheL: MI-guided stochastic morphology-aware tokenizer for pretrained models.

This package implements the MorpheL tokenizer as described in the accompanying
research paper. The canonical pipeline is:

    raw corpus
    -> orthographic words
    -> plausible boundaries
    -> global MI index
    -> top-k candidates
    -> Gumbel cut selection
    -> deterministic cache
    -> contiguous-span vocabulary
    -> native artifacts
    -> downstream adaptation

Public API
----------
The following names are importable from the top-level package:

    from morphel import MorpheLConfig, train_morphel
    from morphel import load_native_tokenizer, encode_sentence_as_tokens

For CLI usage::

    morphel train-tokenizer --config configs/tokenizer/standard/turkish.yaml

Reproducibility
---------------
Set ``seed`` in ``MorpheLConfig`` or pass ``--seed`` to the CLI. The seed
controls NumPy's random state used for Gumbel sampling. Tokenizer induction at
T=0 is fully deterministic given the same corpus, configuration, and package
version.

Language configuration
----------------------
The vowel inventory is **language-dependent**. Always supply the complete vowel
inventory for the target language via ``--vowels`` or the ``language.vowels``
config entry. Do not copy the Turkish vowel inventory to another language
without linguistic justification.
"""

from __future__ import annotations

from morphel.config import FallbackStats, MorpheLConfig, TokenizerMetrics
from morphel.native_tokenizer import (
    encode_sentence_as_tokens,
    encode_word,
    train_morphel,
)
from morphel.serialization import load_native_tokenizer

__version__ = "0.1.0"
__all__ = [
    "MorpheLConfig",
    "TokenizerMetrics",
    "FallbackStats",
    "train_morphel",
    "encode_word",
    "encode_sentence_as_tokens",
    "load_native_tokenizer",
]
