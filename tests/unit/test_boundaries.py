"""Unit tests for boundary heuristics."""

from morphel.boundaries import is_plausible_boundary


def test_is_plausible_boundary() -> None:
    vowels = set("aeiou")

    # Word: "banana"
    # Valid boundaries (pos): 2 (ba|nana), 3 (ban|ana), 4 (bana|na)
    
    # "ba" (ends in vowel) | "na" (starts with consonant)
    # L=ba (has vowel), R=na (no vowel in first 2 chars? Wait, 'na' has 'a' in first two chars)
    # left_window = "ba" (has vowel)
    # right_window = "na" (has vowel)
    # -> left_has_vowel == right_has_vowel -> False (not plausible)
    assert not is_plausible_boundary("banana", 2, vowels)

    # Word: "book"
    # pos 2: L="bo" (vowel), R="ok" (vowel) -> False
    assert not is_plausible_boundary("book", 2, vowels)

    # Let's find a word with a vowel transition.
    # L has vowel, R does not have vowel.
    # e.g., "action". pos 4 (acti|on)
    # R="on" (has vowel), L="ti" (has vowel) -> False
    # e.g., "rhythm" (no vowels)
    assert not is_plausible_boundary("rhythm", 2, vowels)

    # What about "actress"?
    # pos 3: act|ress
    # L="act" (no vowel in last 2 chars -> "ct" -> no vowel)
    # R="re" (has vowel)
    # L(ct)=False, R(re)=True -> True!
    assert is_plausible_boundary("actress", 3, vowels)

def test_is_plausible_boundary_disabled() -> None:
    vowels = set("aeiou")
    assert is_plausible_boundary("banana", 2, vowels, use_filter=False)
    # Still respects the length constraints
    assert not is_plausible_boundary("banana", 1, vowels, use_filter=False)
    assert not is_plausible_boundary("banana", 5, vowels, use_filter=False)
