# MorpheL Makefile
# All commands use the active Python environment.

.PHONY: help install install-dev test lint format typecheck build clean \
        reproduce-smoke reproduce-paper reproduce-tokenizers reproduce-intrinsic \
        reproduce-regime-a reproduce-regime-b reproduce-regime-c \
        reproduce-significance reproduce-tables

PYTHON  ?= python
PYTEST  ?= pytest
PIP     ?= pip
MORPHEL ?= morphel

# Default target
help:
	@echo "MorpheL — available targets"
	@echo ""
	@echo "  install              Install the package (no dev extras)"
	@echo "  install-dev          Install with all dev extras"
	@echo "  test                 Run the full test suite"
	@echo "  lint                 Run ruff linting checks"
	@echo "  format               Apply ruff + black formatting"
	@echo "  typecheck            Run mypy type checks"
	@echo "  build                Build source distribution and wheel"
	@echo "  clean                Remove build and cache artifacts"
	@echo ""
	@echo "  reproduce-smoke      Smoke-mode reproduction (tiny corpus, CPU)"
	@echo "  reproduce-paper      Full paper reproduction (requires GPU + disk)"
	@echo "  reproduce-tokenizers Train tokenizers for all languages"
	@echo "  reproduce-intrinsic  Compute intrinsic tokenizer metrics"
	@echo "  reproduce-regime-a   Run Regime A downstream experiments"
	@echo "  reproduce-regime-b   Run Regime B downstream experiments"
	@echo "  reproduce-regime-c   Run Regime C downstream experiments"
	@echo "  reproduce-significance  Run significance tests"
	@echo "  reproduce-tables     Export paper-ready result tables"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -U pip
	$(PIP) install -e ".[dev,downstream,stats]"

test:
	$(PYTEST) tests/ -v --cov=morphel --cov-report=term-missing

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ -v

test-regression:
	$(PYTEST) tests/regression/ -v

lint:
	ruff check src/ tests/ scripts/ examples/
	black --check src/ tests/ scripts/ examples/

format:
	ruff check --fix src/ tests/ scripts/ examples/
	black src/ tests/ scripts/ examples/

typecheck:
	mypy src/morphel/

build:
	$(PYTHON) -m build

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage

# ---------------------------------------------------------------------------
# Reproduction targets
# ---------------------------------------------------------------------------

# Smoke mode: tiny corpus, CPU-only, verifies software integrity
reproduce-smoke:
	@echo "=== MorpheL smoke reproduction ==="
	@echo "This verifies software integrity, not paper scores."
	@echo ""
	$(MORPHEL) train-tokenizer \
		--dataset facebook/xnli \
		--subset tr \
		--language tr \
		--vowels "aeiouAEIOUıİöüÖÜâîûÂÎÛ" \
		--splits train \
		--eval-splits validation \
		--text-columns premise hypothesis \
		--output-dir outputs/smoke/morphel-tr \
		--vocab-size 1000 \
		--min-frequency 1 \
		--top-k 4 \
		--temperature 1.0 \
		--mi-threshold 0.0 \
		--seed 42
	$(MORPHEL) inspect-tokenizer \
		--tokenizer-dir outputs/smoke/morphel-tr
	$(MORPHEL) encode \
		--tokenizer-dir outputs/smoke/morphel-tr \
		--text "Merhaba, bu bir test cümlesidir."
	@echo "=== Smoke test complete ==="

# Full paper reproduction — requires GPU and significant disk space
reproduce-paper: reproduce-tokenizers reproduce-intrinsic \
                 reproduce-regime-a reproduce-regime-b reproduce-regime-c \
                 reproduce-significance reproduce-tables
	@echo "=== Paper reproduction complete ==="
	@echo "See results/paper_tables/ for aggregated outputs."

reproduce-tokenizers:
	bash scripts/reproduce_tokenizers.sh

reproduce-intrinsic:
	bash scripts/reproduce_intrinsic_metrics.sh

reproduce-regime-a:
	bash scripts/reproduce_regime_a.sh

reproduce-regime-b:
	bash scripts/reproduce_regime_b.sh

reproduce-regime-c:
	bash scripts/reproduce_regime_c.sh

reproduce-significance:
	bash scripts/reproduce_significance.sh

reproduce-tables:
	bash scripts/reproduce_tables.sh
