"""Unit tests for MI index building."""

from morphel.mi import build_global_mi_index


def test_build_global_mi_index() -> None:
    # A tiny corpus representation
    word_freq = {
        "pretest": 10,
        "preamble": 5,
        "test": 20,
        "amble": 10,
        "predict": 15,
        "dict": 5,
    }
    vowels = set("aeiou")
    
    mi_index = build_global_mi_index(
        word_freq,
        vowels,
        mi_threshold=0.0,
        use_vowel_boundary_filter=False, # Disable for simpler test logic
    )
    
    # We expect some boundaries for 'pretest'.
    # For example, "pre" + "test".
    assert "pretest" in mi_index
    # The returned list is sorted by descending MI.
    # We just ensure it's a list of (position, score) tuples.
    assert isinstance(mi_index["pretest"], list)
    if mi_index["pretest"]:
        pos, score = mi_index["pretest"][0]
        assert isinstance(pos, int)
        assert isinstance(score, float)
