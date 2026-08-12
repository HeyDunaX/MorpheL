# When Morphology Matters: MorpheL for Tokenizer Adaptation under Constrained Adaptation

A novel tokenization toolkit that leverages mutual information between morphological boundaries for vocabulary induction and applies Gumbel-max sampling to induce stochastic segmentation diversity during training.

> This repository accompanies the paper:  
> **When Morphology Matters: MorpheL for Tokenizer Adaptation under Constrained Adaptation**

---

## Why MorpheL?

Standard subword tokenizers (like BPE and Unigram) are driven purely by data frequency, completely ignoring linguistic morphology. This causes them to shatter morphologically rich languages (MRLs) into semantically meaningless chunks. MorpheL removes this limitation by guiding the vocabulary building process with statistical mutual information (MI) between morphological boundaries, and introduces a stochastic sampling mechanism to expose language models to diverse segmentations during training.

- Unified CLI for preprocessing, vocabulary induction, MI indexing, and intrinsic evaluation.
- Dataset-agnostic schema — plug in any Hugging Face dataset.
- Seamless integration with Hugging Face tokenizers via a WordLevel wrapper.
- Full pipeline: normalization $\rightarrow$ word frequency collection $\rightarrow$ MI index building $\rightarrow$ vocabulary induction $\rightarrow$ T=0 caching $\rightarrow$ evaluation.

---

## Core Features

MorpheL covers the entire tokenizer adaptation lifecycle — from corpus frequency analysis to MI-based span extraction and deterministic/stochastic segmentation.

**MI-Guided Vocabulary Induction**
- Leverages global Mutual Information (MI) index to identify true morphological boundaries.
- Character alphabet coverage guaranteed via overflow appending to prevent OOV issues.

**Stochastic Segmentation**
- Gumbel-max sampling for diverse, morphology-aware subword segmentations during training (`T > 0`).
- Deterministic, maximum-likelihood segmentation during inference (`T = 0`), backed by a highly optimized segmentation cache.

**Evaluation and Reporting**
- Intrinsic evaluation metrics out-of-the-box: fertility, tokens-per-character, average sequence length, vocabulary coverage, and character shatter rate.
- Automated output reports (`morphel_metrics_report.json`).

**Reusability and Extensibility**
- Highly modularized architecture for preprocessing, vocabulary building, and inference.
- Saves native artifacts (`morphel_vocab.json`, `mi_index.pkl`, `segmentation_cache.pkl`) alongside a Hugging Face compatible wrapper.

---

## Command-Line Interface (CLI)

All functionality is exposed through a single command, `morphel train-tokenizer`. You can run `morphel train-tokenizer --help` at any time to see the full options.

```
Usage: morphel train-tokenizer [OPTIONS]

  Train a MorpheL tokenizer on a Hugging Face dataset.

Options:
  --dataset TEXT           Hugging Face dataset identifier (default: facebook/xnli)
  --subset TEXT            Dataset subset or language code, e.g. tr, ru, fr, vi
  --language TEXT          Language identifier stored in morphel_config.json
  --vowels TEXT            Complete target-language vowel inventory. Example for Turkish: aeiouAEIOUıİöüÖÜâîûÂÎÛ
  --splits TEXT            Explicit tokenizer-induction splits. Use 'train' for strict induction.
  --eval-splits TEXT       Splits used only for intrinsic tokenizer evaluation.
  --text-columns TEXT      Dataset columns containing raw text.
  --vocab-size INTEGER     Target nominal vocabulary size.
  --min-frequency INTEGER  Minimum word frequency for vocabulary inclusion.
  --top-k INTEGER          Maximum number of segmentation candidates per word.
  --temperature FLOAT      Gumbel softmax temperature. T=0 means deterministic.
  --mi-threshold FLOAT     Minimum Mutual Information threshold for boundaries.
  --output-dir TEXT        Output directory for saving the native artifacts.
  --model-max-length INTEGER Maximum sequence length for the HF config.
  --seed INTEGER           Random seed for reproducibility.
  --dry-run                Validate inputs without running the pipeline.
  --help                   Show this message and exit.
```

---

## Installation

MorpheL requires Python 3.9+ and can be installed directly from the repository.

```bash
git clone https://github.com/HeyDunaX/MorpheL.git
cd morphel
pip install -e .
```

---

## Quickstart

### Train a Tokenizer

The example below trains a MorpheL tokenizer on the Turkish subset of XNLI, using the standard vowels for Turkish.

```bash
morphel train-tokenizer \
    --language tr \
    --vowels "aeiouAEIOUıİöüÖÜâîûÂÎÛ" \
    --splits train \
    --output-dir outputs/tokenizers/morphel-tr
```

### Loading and Using

Because Hugging Face's `AutoTokenizer` does not natively support MorpheL's segmentation algorithm on raw strings, you must load the native artifacts and pre-tokenize your text before passing it to the model.

```python
from morphel.serialization import load_native_tokenizer
from morphel.preprocessing import simple_tokenize

# Load native artifacts and the fast wrapper
native_vocab, mi_index, seg_cache, hf_tokenizer = load_native_tokenizer("outputs/tokenizers/morphel-tr")

# Encode text
text = "Bu bir Türkçe cümledir."
words = simple_tokenize(text)
# Segment words using native MorpheL logic here...
```

---

## Outputs

Each run writes its results to the specified `--output-dir`.

```
outputs/tokenizers/morphel-tr/
├── morphel_vocab.json
├── mi_index.pkl
├── segmentation_cache.pkl
├── morphel_config.json
├── morphel_metrics_report.json
├── tokenizer.json
└── tokenizer_config.json
```

---

## Code Structure

The source code lives entirely under `src/morphel/`. Each module has a single, well-defined responsibility.

```
src/morphel/
├── cli.py               # CLI and argument parsing
├── native_tokenizer.py  # Orchestrator for training and inference
├── preprocessing.py     # Tokenization and frequency analysis
├── mi.py                # Mutual Information index building
├── vocabulary.py        # Morphology-aware vocabulary induction
├── boundaries.py        # Boundary classification
├── segmentation.py      # Core segmentation logic
├── sampling.py          # Gumbel-max sampling
├── coalescing.py        # Subword piece coalescing
├── fallback.py          # Recursive fallback for OOV words
├── metrics.py           # Intrinsic evaluation metrics
├── serialization.py     # Saving and loading native artifacts
└── config.py            # Hyperparameter dataclasses
```

---

## Reproducibility

MorpheL is designed to guarantee full deterministic reproducibility.

- Fixed random seeds for sampling and deterministic tie-breaking.
- Caching of T=0 deterministic maximum-likelihood boundaries.
- Dependency pinning via `pyproject.toml` and `uv.lock`.

---

## License

MorpheL is released as open-source software under the MIT License.

---

## Citation

```bibtex
@inproceedings{morphel,
    title = "{When Morphology Matters: MorpheL for Tokenizer Adaptation under Constrained Adaptation}",
    author = "",
    booktitle = "",
    year = ""
}
```

## References

1. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., Stoyanov, V.: Unsupervised cross-lingual representation learning at scale. In: Proceedings of the 58th annual meeting of the association for computational linguistics. pp. 8440–8451 (2020)
2. Conneau, A., Rinott, R., Lample, G., Williams, A., Bowman, S.R., Schwenk, H., Stoyanov, V.: Xnli: Evaluating cross-lingual sentence representations. In: Proceedings of the 2018 conference on empirical methods in natural language processing. pp. 2475–2485 (2018)
3. Creutz, M., Lagus, K.: Unsupervised models for morpheme segmentation and morphology learning. ACM Transactions on Speech and Language Processing (TSLP) 4(1), 1–34 (2007)
4. Ercan, G., Yildiz, O.T.: Grammar or crammer? the role of morphology in distinguishing orthographically similar but semantically unrelated words. IEEE Access 13, 64412–64458 (2025). https://doi.org/10.1109/ACCESS.2025.3557086
5. Greenberg, J.H.: A quantitative approach to the morphological typology of language. International Journal of American Linguistics 26(3), 178–194 (1960). https://doi.org/10.1086/464575
6. Kudo, T.: Subword regularization: Improving neural network translation models with multiple subword candidates. In: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 66–75 (2018)
