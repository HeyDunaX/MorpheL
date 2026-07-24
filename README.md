# MorpheL Tokenizer

Anonymous artifact repository for the MorpheL research paper. 

**MorpheL** (MI-guided stochastic morphology-aware tokenizer) is a novel tokenization algorithm that builds vocabulary based on mutual information between morphological boundaries, and applies Gumbel-max sampling to induce stochastic segmentation diversity during training.

## Installation

This repository uses modern Python packaging via `uv` or `pip`.

```bash
# Clone the repository
git clone https://github.com/anonymous/morphel.git
cd morphel

# Install using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

## Quick Start

Train a MorpheL tokenizer on the Turkish XNLI subset:

```bash
morphel train-tokenizer \
    --language tr \
    --vowels "aeiouAEIOUıİöüÖÜâîûÂÎÛ" \
    --splits train validation test \
    --output-dir outputs/tokenizers/morphel-tr
```

For detailed configuration, see the `configs/` directory.

## Repository Structure

- `src/morphel/`: Core tokenizer algorithms and downstream model wrappers.
- `configs/`: Standard YAML configurations for reproducing the paper's results.
- `tests/`: Extensive unit, integration, and regression test suites.
- `metadata/`: Structured metadata for artifact evaluation.

## License

Pending final publication. See `LICENSE_PENDING.md`.
