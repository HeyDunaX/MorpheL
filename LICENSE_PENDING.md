# License Decision Required

The project owner has not yet supplied the final software license for the
MorpheL repository.

## Action required

Before public release, choose and add a license compatible with:

- the intended open-source distribution policy;
- Hugging Face and PyPI publication plans;
- all third-party dependencies listed in `pyproject.toml`;
- the paper venue's artifact policy for accepted papers.

## Common choices for NLP research repositories

| License | Notes |
|---|---|
| MIT | Permissive; widely used in NLP research |
| Apache 2.0 | Permissive + patent grant; Hugging Face default |
| CC BY 4.0 | Common for model weights; less suitable for code |

## How to add the license

1. Create a `LICENSE` file in the repository root.
2. Replace this file (`LICENSE_PENDING.md`) with `LICENSE`.
3. Update `pyproject.toml`: `license = { file = "LICENSE" }`.
4. Update `CITATION.cff`: `license: <SPDX-identifier>`.
5. Update `codemeta.json`: `"license": "https://spdx.org/licenses/<ID>.html"`.
6. Update `metadata/paper.yaml`: `license: <SPDX-identifier>`.

## Third-party dependency licenses

| Package | License |
|---|---|
| numpy | BSD 3-Clause |
| tqdm | MIT + MPLv2 |
| PyYAML | MIT |
| datasets | Apache 2.0 |
| tokenizers | Apache 2.0 |
| transformers | Apache 2.0 |
| huggingface-hub | Apache 2.0 |
| typer | MIT |
| rich | MIT |
| torch | BSD 3-Clause |
| scipy | BSD 3-Clause |
| scikit-learn | BSD 3-Clause |
| statsmodels | BSD 3-Clause |

> **Note:** Do not redistribute XNLI data, XLM-R weights, or fastText
> embeddings. These are downloaded at runtime via their respective
> distribution channels and are subject to their own licenses.

Do not label the repository as MIT, Apache-2.0, or any other specific license
until this decision has been confirmed by the project owner.
