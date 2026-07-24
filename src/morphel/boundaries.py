"""Candidate intra-word boundary identification for MorpheL.

The MorpheL vowel-transition heuristic identifies positions where the presence
of a vowel changes between the two-character windows on either side of the
boundary. This heuristic is **language-dependent**.

Language-specific requirements
-------------------------------
- ``vowels`` must be the complete orthographic vowel inventory of the target
  language. Include uppercase variants if the corpus is not lowercased. Include
  vowels with diacritics when they appear in the writing system.
- Do not copy the Turkish vowel inventory to another language without
  linguistic justification.
- For writing systems where this heuristic is not meaningful (e.g., Chinese),
  set ``use_filter=False`` and report that ablation explicitly.

Boundary position constraints
-------------------------------
A valid boundary position ``i`` satisfies ``2 <= i <= len(word) - 2``, which
ensures both the prefix ``word[:i]`` and suffix ``word[i:]`` have at least two
characters.
"""

from __future__ import annotations


def is_plausible_boundary(
    word: str,
    position: int,
    vowels: set[str],
    use_filter: bool = True,
) -> bool:
    """Apply the MorpheL vowel-transition boundary heuristic.

    A boundary at ``position`` is eligible when the two-character window
    before the boundary and the two-character window after the boundary differ
    in whether they contain a target-language vowel.

    Args:
        word: The orthographic word string.
        position: Candidate boundary position (character index). Valid range is
            ``[2, len(word) - 2]``.
        vowels: Set of vowel characters for the target language. This is
            language-dependent; supply the complete inventory.
        use_filter: If False, all positions in the valid range are accepted
            without applying the vowel-transition test. Set to False only when
            linguistically justified and report as an ablation.

    Returns:
        True if the position is a plausible boundary, False otherwise.

    Determinism:
        This function is fully deterministic.

    Language assumption:
        The heuristic is designed for alphabetic scripts with vowel/consonant
        alternation. It is not meaningful for logographic systems (Chinese) or
        scripts where vowels are implied rather than written (some abjads).
    """
    if position < 2 or position > len(word) - 2:
        return False
    if not use_filter:
        return True

    left_window = word[max(0, position - 2) : position]
    right_window = word[position : position + 2]
    left_has_vowel = any(character in vowels for character in left_window)
    right_has_vowel = any(character in vowels for character in right_window)
    return left_has_vowel != right_has_vowel
