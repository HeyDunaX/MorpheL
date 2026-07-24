# Changelog

All notable changes to MorpheL are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0-review] — Anonymous Review Artifact

This is the initial public release of the MorpheL repository, prepared as an
anonymous artifact for peer review. The paper has not yet been accepted or
published. Publication metadata will be updated after acceptance.

### Added

**Core tokenizer package (`src/morphel/`)**

- `MorpheLConfig` — frozen dataclass for all algorithm parameters with
  `validate()` pre-flight checks.
- `build_global_mi_index` — corpus-global prefix/suffix mutual-information
  index with token-frequency weighting and positive-MI filtering.
- `gumbel_sample` — Gumbel-max sampling with deterministic argmax at T≤0.
- `segment_word` — zero/one/two-cut segmentation under the canonical gamma
  formulation.
- `build_segmentation_cache` — deterministic T=0 segmentation cache.
- `count_contiguous_spans` — contiguous-span vocabulary induction.
- `build_vocabulary` — nominal vocabulary + character-coverage overflow.
- `recursive_fallback` — recursive longest-match OOV decomposition.
- `coalesce_pieces` — DP-optimal vocabulary coalescing.
- `encode_word` / `encode_sentence_as_tokens` — native MorpheL encoding path.
- `build_fast_tokenizer` / `save_native_artifacts` — Hugging Face wrapper and
  native artifact serialization.
- `compute_metrics` — intrinsic fertility, OOV, and fallback metrics.

**CLI (`morphel` entry point)**

- `train-tokenizer` — full pipeline with YAML config + CLI override support.
- `inspect-tokenizer` — report vocabulary and config details.
- `encode` — encode raw text through native MorpheL artifacts.
- `evaluate-intrinsic` — compute metrics on existing tokenizer.
- `validate-artifacts` — verify native artifact integrity.
- `push-to-hub` — upload tokenizer directory to Hugging Face Hub.
- `run-regime-a/b/c` — downstream adaptation stubs.
- `significance` — statistical significance testing.
- `export-results` — aggregate results and export tables.

**Variants (paper-specific; not universal defaults)**

- `morphel.variants.stochastic_consensus` — Russian stochastic-consensus v4.
- `morphel.variants.lexical_anchor` — French lexical-anchor v9.

**Statistical testing (`src/morphel/statistics/`)**

- Exact McNemar test for accuracy.
- Stratified paired bootstrap (10,000 samples default).
- Paired randomization tests.
- Holm correction over primary metrics.
- Confounding warnings for seed and regime mismatches.

**Configurations**

- Language configs for Turkish, Russian, French, Vietnamese, Swahili, Chinese.
- Downstream configs for Regimes A, B, C from the paper protocol.
- Significance config with default resampling counts.

**Tests**

- Unit tests for all core algorithmic components.
- Integration tests for train → save → load → encode round-trips.
- Regression tests against golden fixtures from the canonical implementation.

**Documentation**

- `README.md` — paper-oriented; 25 sections; anonymous.
- `docs/` — algorithm, multilingual adaptation, native artifacts, downstream
  regimes, statistical testing, reproducibility, limitations, adding a language,
  paper results, and variant documentation.
- `AUDIT.md` — pre-refactor audit of the canonical implementation.
- `RELEASE_CHECKLIST.md` — pre-publication verification checklist.

### Known limitations

- Regime A/B/C downstream implementations are stubs pending the project
  owner's supply of canonical code.
- WECHSEL embedding initialization is an interface stub.
- Vietnamese vowel inventory requires verification by the project owner.
- Chinese requires an explicit word-segmentation policy; the vowel filter is
  disabled.
- License is pending the project owner's decision.
- All Hugging Face artifact repository identifiers are `null` pending anonymous
  mirror creation.

---

<!-- Add future versions above this line -->
