import json
import os
import sys
from collections import Counter
from pathlib import Path

# Mock ML libraries that fail to load DLLs on Windows due to App Control policies
from unittest.mock import MagicMock
sys.modules["datasets"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["tokenizers"] = MagicMock()
sys.modules["huggingface_hub"] = MagicMock()
sys.modules["pandas"] = MagicMock()

# Add handoff directory to path so we can import the canonical implementation
handoff_dir = Path("d:/Repository/morphel/morphel_agent_handoff_anonymous").resolve()
sys.path.insert(0, str(handoff_dir))

from morphel_tokenizer import (
    build_global_mi_index,
    build_segmentation_cache,
    build_vocabulary,
    collect_word_frequencies,
    count_contiguous_spans,
    build_character_alphabet,
)

def main():
    fixtures_dir = Path("d:/Repository/morphel/tests/fixtures").resolve()
    corpus_path = fixtures_dir / "tiny_corpus.txt"
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(sentences)} sentences.")
    
    # 1. Collect word frequencies
    word_freq = collect_word_frequencies(sentences)
    
    # Turkish vowels as a broad superset for this test
    vowels = set("aeiouAEIOUıİöüÖÜâîûÂÎÛéèêëàâîïôùûüÿœŒаеёиоуыэюяАЕЁИОУЫЭЮЯ")
    
    # 2. Build MI index
    mi_index = build_global_mi_index(word_freq, vowels, mi_threshold=0.0)
    
    # 3. Build segmentation cache
    # Seed numpy randomly to a fixed value
    import numpy as np
    import random
    random.seed(42)
    np.random.seed(42)
    
    cache = build_segmentation_cache(word_freq.keys(), mi_index, top_k=4)
    
    # 4. Count spans
    spans = count_contiguous_spans(word_freq, cache)
    
    # 5. Build vocab
    vocab, _ = build_vocabulary(
        subword_frequency=spans,
        character_alphabet=build_character_alphabet(word_freq.keys()),
        vocab_size=100,
        min_frequency=1,
    )
    
    # Save outputs
    with open(fixtures_dir / "golden_segmentations.json", "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        
    with open(fixtures_dir / "tiny_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
        
    print("Golden fixtures generated successfully.")

if __name__ == "__main__":
    main()
