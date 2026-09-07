# Mind the Morphology: Morphology-Aware Tokenizer Adaptation under Constrained Fine-Tuning
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A morphology-aware tokenization method that uses corpus-level mutual information to rank candidate prefix--suffix decompositions and Gumbel-max sampling to control segmentation granularity.

> This repository accompanies the paper:  
> **Mind the Morphology: Morphology-Aware Tokenizer Adaptation under Constrained Fine-Tuning**

> Nhan Tran, Hung To, Nguyen Thi Tuyet Hai. Submission to SoICT 2026

---

## Why MorpheL?

Standard subword tokenizers such as BPE and Unigram optimize corpus
statistics, but do not explicitly model morphological structure. In
morphologically rich languages, this may produce fragmented and inconsistent
representations of related word forms. MorpheL introduces a morphology-aware
vocabulary induction pipeline that ranks structurally plausible
prefix--suffix decompositions with corpus-level mutual information (MI).
Gumbel-max sampling then provides controlled variation in segmentation
granularity, while deterministic $T=0$ caching ensures reproducible runtime
encoding.

---

## Core Features

MorpheL provides an end-to-end tokenizer adaptation pipeline, from corpus
preprocessing and MI indexing to vocabulary induction, deterministic encoding,
and intrinsic evaluation.

**MI-Guided Vocabulary Induction**
- Uses a corpus-level Mutual Information (MI) index to rank structurally
  plausible prefix--suffix decompositions.
- Builds a fixed-budget vocabulary from cached segmentation pieces and
  contiguous piece spans.
- Appends missing corpus characters to ensure complete character coverage.

**Controlled Segmentation**
- Uses Gumbel-max sampling (`T > 0`) to explore alternative cut counts over
  MI-ranked candidate positions.
- Uses deterministic `T = 0` selection for persistent vocabulary construction
  and runtime encoding.
- Caches deterministic segmentations for reproducible encoding.

**Evaluation and Reporting**
- Provides intrinsic metrics including fertility, tokens-per-character,
  average token length, vocabulary coverage, and character shatter rate.
- Produces machine-readable evaluation reports
  (`morphel_metrics_report.json`).

**Reusability and Extensibility**
- Modular components for preprocessing, candidate generation, MI indexing,
  vocabulary construction, caching, and native encoding.
- Saves reproducibility artifacts including
  `morphel_vocab.json`, `mi_index.pkl`, and `segmentation_cache.pkl`.
- Provides a Hugging Face-compatible wrapper for downstream integration.

<img width="1344" height="864" alt="MorpheL" src="https://github.com/user-attachments/assets/0966c21a-096c-4108-aed6-59209fe15cbb" />

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

## Sample Results

MorpheL demonstrates its strongest benefits under constrained adaptation,
particularly for morphologically rich languages. The results below summarize
the main findings reported in the paper.

### XNLI Results

We evaluate MorpheL on six XNLI languages: Turkish, Russian, French,
Vietnamese, Swahili, and Chinese. All methods use the same nominal 32K
vocabulary budget and downstream evaluation protocol.

#### Regime A — Classifier-Only Adaptation

MorpheL achieves statistically significant Macro-F1 improvements over the
strongest fixed-checkpoint baseline in three morphologically rich languages:

| Language | IS | Comparator | Δ Accuracy | Δ Macro-F1 | p-value |
|----------|----:|------------|------------:|-----------:|--------:|
| Russian  | 2.30 | WordPiece | **+1.40** | **+2.61** | **0.0014** |
| Swahili  | 2.55 | WordPiece | **+0.52** | **+4.86** | **<0.001** |
| Turkish  | 2.86 | WordPiece | **+1.40** | **+4.07** | **<0.001** |

For comparison, the remaining languages show smaller or non-significant
differences under the same regime:

| Language | IS | Comparator | Δ Accuracy | Δ Macro-F1 | p-value |
|----------|----:|------------|------------:|-----------:|--------:|
| Vietnamese | 1.06 | BPE | +0.32 | +1.36 | 0.090 |
| Chinese    | 1.15 | BPE | -1.04 | -0.57 | 0.574 |
| French     | 1.45 | Unigram | -0.14 | +0.30 | 0.658 |

Positive values favor MorpheL.

### Cross-Regime Results

MorpheL's advantage is not limited to the classifier-only setting. Under
Regime B, where replacement embeddings and the classification head are
updated, MorpheL remains ahead of the strongest baseline in Turkish, Russian,
and Swahili:

| Language | Regime A Δ Macro-F1 | Regime B Δ Macro-F1 | Regime C Δ Macro-F1 |
|----------|--------------------:|--------------------:|--------------------:|
| Turkish  | **+2.80** | **+2.42** | -1.42 |
| Russian  | **+1.23** | **+0.91** | -1.76 |
| Swahili  | **+3.01** | **+3.48** | -1.26 |

The gains largely disappear under Regime C (full-model fine-tuning), where
conventional tokenizers can recover or surpass MorpheL.

### MorphBPE Comparison

MorpheL is also compared directly with the morphology-aware MorphBPE baseline
on Turkish and Swahili under Regimes A and B.

| Regime | Language | MorphBPE Macro-F1 | MorpheL Macro-F1 | Gain |
|--------|----------|------------------:|-----------------:|------:|
| A | Turkish | 32.60 ± 2.00 | **36.04 ± 0.21** | **+3.44** |
| A | Swahili | 29.52 ± 1.92 | **32.53 ± 1.85** | **+3.01** |
| B | Turkish | 32.63 ± 2.11 | **36.06 ± 0.55** | **+3.43** |
| B | Swahili | 29.20 ± 1.57 | **32.68 ± 1.42** | **+3.48** |

MorpheL achieves higher Accuracy and Macro-F1 than MorphBPE in all four
evaluated regime-language combinations.

### Fragmentation Results

Relative to WordPiece, MorpheL reduces both tokens-per-character and
fertility across the five evaluated whitespace-delimited languages:

| Language | Δ Tokens/char | Δ Fertility | Δ Avg. Length |
|----------|--------------:|------------:|--------------:|
| Turkish    | -5.19% | -5.21% | -5.19% |
| Russian    | -5.81% | -5.83% | +5.19% |
| French     | -4.44% | -4.21% | -4.21% |
| Vietnamese | -7.13% | -7.12% | -7.11% |
| Swahili    | -7.70% | -7.70% | -7.72% |

Overall, MorpheL reduces fragmentation by approximately 4%–8% on the
evaluated languages.

### Ablation Results

A Turkish Regime-A ablation examines the contribution of MI-guided ranking
and the Top-k configuration:

| Variant | Accuracy | Macro-F1 | Setting |
|---------|---------:|---------:|---------|
| Random boundaries | 36.05 ± 0.31 | 31.93 ± 2.55 | No MI |
| MI only | 37.96 ± 0.44 | 35.95 ± 1.01 | T = 0 |
| Top-k = 1 | 36.77 ± 0.47 | 32.32 ± 1.90 | MI–Gumbel |
| **Full MorpheL** | **38.26 ± 0.32** | **36.04 ± 0.21** | **k = 4** |
| Top-k = 8 | 36.95 ± 0.46 | 34.59 ± 0.90 | MI–Gumbel |

The MI-only configuration approaches the full MorpheL result, while random
boundary selection performs substantially worse. Among the tested candidate
budgets, `k = 4` gives the highest Macro-F1.

### Generalization to Machine Translation

To assess transfer beyond sentence classification, MorpheL is evaluated as
the Turkish target tokenizer in a controlled English–Turkish Transformer
experiment on OPUS-100.

| Target Tokenizer | BLEU |
|------------------|-----:|
| BPE       | 15.04 |
| WordPiece | 12.71 |
| Unigram   | 15.16 |
| **MorpheL** | **16.24** |

MorpheL achieves the highest BLEU, with improvements of:

- **+1.20 BLEU** over BPE
- **+3.53 BLEU** over WordPiece
- **+1.08 BLEU** over Unigram

Each MT system was run once, so these results are reported as controlled
generalization evidence rather than statistically conclusive superiority.

### Qualitative Segmentation

Representative examples illustrate the reusable units produced by MorpheL:

```text
Turkish
Sana | gönd | erme | den | önce | bazı | habe | rler | ald | ım | ...

Swahili
Watoto | walikuwa | waki | cheza | uwanjani | baada | ya | shule
```
---

## Dataset

MorpheL is evaluated on the **Cross-lingual Natural Language Inference
(XNLI)** benchmark.

We use the **Turkish, Russian, French, Vietnamese, Swahili, and Chinese**
subsets of XNLI to cover a range of morphological profiles.

Tokenizer induction uses only the **premise and hypothesis text from the
corresponding target-language training split**. Validation and test text, as
well as all labels, are excluded from tokenizer construction. All tokenizers
use the same nominal **32K vocabulary budget** and identical downstream
evaluation splits.

---

## Data Format

MorpheL supports datasets loaded through the Hugging Face `datasets` library.
For tokenizer induction, the input dataset must provide one or more text fields
containing the target-language data.

For XNLI, MorpheL uses the `premise` and `hypothesis` fields from the
target-language training split:

```json
{
  "premise": "A person is riding a bicycle.",
  "hypothesis": "A person is outdoors.",
  "label": 0
}

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

## Ethics Statement

This work uses existing, publicly available benchmark data and does not collect
new data from human participants. The XNLI experiments use the corresponding
language subsets and training splits of XNLI, while the machine translation
experiment uses the English--Turkish configuration of OPUS-100.

Tokenizer induction uses only target-language training text; validation and test
text and evaluation labels are excluded from tokenizer construction. No
personally identifiable information is collected, inferred, or used as part of
the proposed method.

As with other multilingual benchmarks, the results may reflect biases,
limitations, and coverage differences present in the underlying datasets and
pretrained language resources. We therefore interpret the reported gains within
the evaluated languages and tasks rather than as evidence of universal
morphological or linguistic behavior.

Users of the released implementation should follow the original licenses and
usage requirements of the datasets, pretrained models, and third-party
software components on which the experiments depend.

---
## License

MorpheL is released as open-source software under the **MIT License**.

The MIT License permits use, copying, modification, merging, publishing,
distribution, sublicensing, and sale of copies of the software, subject to
the conditions specified in the license.

The software is provided **"as is"**, without warranty of any kind. The full
license terms are available in the [`LICENSE`](LICENSE) file.

### Third-Party Components

MorpheL may depend on or integrate with third-party libraries, datasets, and
pretrained models that are distributed under their own licenses. These
components remain subject to their respective license terms.

Users are responsible for reviewing and complying with the licenses and usage
conditions of any third-party dependencies, datasets, models, and external
resources used with MorpheL.

---

## Citation

```bibtex
@inproceedings{morphel,
    title = "{Mind the Morphology: Morphology-Aware Tokenizer Adaptation under Constrained Fine-Tuning}",
    author = "Nhan Tran, Hung To and Nguyen Thi Tuyet Hai",
    booktitle = "{SoICT Submission}",
    year = "{2026}"
}
```

## References

1. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., Stoyanov, V.: Unsupervised cross-lingual representation learning at scale. In: Proceedings of the 58th annual meeting of the association for computational linguistics. pp. 8440–8451 (2020)
2. Conneau, A., Rinott, R., Lample, G., Williams, A., Bowman, S.R., Schwenk, H., Stoyanov, V.: Xnli: Evaluating cross-lingual sentence representations. In: Proceedings of the 2018 conference on empirical methods in natural language processing. pp. 2475–2485 (2018)
3. Creutz, M., Lagus, K.: Unsupervised models for morpheme segmentation and morphology learning. ACM Transactions on Speech and Language Processing (TSLP) 4(1), 1–34 (2007)
4. Ercan, G., Yildiz, O.T.: Grammar or crammer? the role of morphology in distinguishing orthographically similar but semantically unrelated words. IEEE Access 13, 64412–64458 (2025). https://doi.org/10.1109/ACCESS.2025.3557086
5. Greenberg, J.H.: A quantitative approach to the morphological typology of language. International Journal of American Linguistics 26(3), 178–194 (1960). https://doi.org/10.1086/464575
6. Kudo, T.: Subword regularization: Improving neural network translation models with multiple subword candidates. In: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 66–75 (2018)
