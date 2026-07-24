"""MorpheL word segmentation and deterministic cache construction.

The MorpheL formulation supports zero, one, or two cuts per word. The number
of cuts is sampled from a categorical distribution whose logits (``gamma``) are
derived from the top-k MI scores.

Gamma formulation (canonical)
------------------------------
Given the top-k MI candidates sorted by descending score::

    gamma[0] = 0.0                              # zero cuts
    gamma[1] = mi_scores[0]                     # one cut
    gamma[2] = (mi_scores[0]+mi_scores[1]) / 2  # two cuts (or -1e9 if k<2)

The number of cuts is sampled with :func:`morphel.sampling.gumbel_sample`.
At T=0 this reduces to argmax, which always picks the option with the highest
logit.

Deterministic cache
--------------------
:func:`build_segmentation_cache` calls :func:`segment_word` at T=0 for every
word type in the corpus. The resulting cache is serialized as a native
artifact and used for all vocabulary induction and evaluation. This makes the
tokenizer's segmentation fully reproducible given the same MI index and
configuration.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from tqdm.auto import tqdm

from morphel.sampling import gumbel_sample
from morphel.typing import Boundary, MIIndex, SegmentationCache


def segment_word(
    word: str,
    mi_index: Mapping[str, Sequence[Boundary]],
    top_k: int = 4,
    temperature: float = 1.0,
) -> list[str]:
    """Segment one word with the MorpheL MI-plus-Gumbel procedure.

    The current MorpheL formulation supports zero, one, or two cuts. Candidate
    boundary locations come exclusively from the global positive-MI index.

    Args:
        word: The orthographic word string to segment.
        mi_index: Global MI index mapping word strings to candidate boundaries.
        top_k: Maximum number of top-MI candidates to consider.
        temperature: Gumbel sampling temperature. Use 0.0 for deterministic
            segmentation (vocabulary induction and evaluation).

    Returns:
        List of piece strings. Returns ``[word]`` when the word has no
        plausible boundaries or fewer than 3 characters.

    Determinism:
        At temperature <= 0.0 this function is fully deterministic.
        At temperature > 0.0 the result depends on NumPy's random state.

    Note:
        Special tokens of the form ``<token>`` are returned unsegmented.
    """
    if word.startswith("<") and word.endswith(">"):
        return [word]
    if len(word) <= 2:
        return [word]

    candidates = list(mi_index.get(word, ()))
    if not candidates:
        return [word]

    # Sort by descending MI score; keep top-k
    candidates.sort(key=lambda item: item[1], reverse=True)
    candidates = candidates[:top_k]
    mi_scores = [score for _, score in candidates]

    # Canonical gamma formulation — zero / one / two cuts
    gamma = [
        0.0,
        float(mi_scores[0]),
        (
            float((mi_scores[0] + mi_scores[1]) / 2.0)
            if len(mi_scores) >= 2
            else -1e9
        ),
    ]
    number_of_cuts = gumbel_sample(gamma, temperature)
    if number_of_cuts == 0:
        return [word]

    selected_positions = sorted(position for position, _ in candidates[:number_of_cuts])
    pieces: list[str] = []
    start = 0
    for position in selected_positions:
        pieces.append(word[start:position])
        start = position
    pieces.append(word[start:])
    return [piece for piece in pieces if piece]


def build_segmentation_cache(
    word_types: Iterable[str],
    mi_index: MIIndex,
    top_k: int,
) -> SegmentationCache:
    """Build the deterministic T=0 segmentation cache for vocabulary induction.

    The cache is built by calling :func:`segment_word` at temperature 0.0
    (deterministic argmax) for every word type in the corpus, sorted
    lexicographically for a consistent ordering.

    Args:
        word_types: Iterable of word type strings to cache.
        mi_index: Global MI index.
        top_k: Maximum number of top-MI candidate boundaries per word.

    Returns:
        A :class:`~morphel.typing.SegmentationCache` mapping every word string
        to its deterministic list of piece strings.

    Determinism:
        Fully deterministic. No random state is consumed.
    """
    return {
        word: segment_word(word, mi_index, top_k=top_k, temperature=0.0)
        for word in tqdm(sorted(set(word_types)), desc="Caching T=0 segmentations")
    }
