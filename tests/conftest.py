"""Pytest configuration and shared fixtures.

Provides access to the golden output fixtures generated from the original
canonical implementation.
"""

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def tiny_corpus() -> list[str]:
    """Load the tiny corpus sentences."""
    with (FIXTURES_DIR / "tiny_corpus.txt").open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@pytest.fixture(scope="session")
def golden_segmentations() -> dict[str, list[str]]:
    """Load the golden T=0 segmentations."""
    with (FIXTURES_DIR / "golden_segmentations.json").open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def tiny_vocab() -> dict[str, int]:
    """Load the golden vocabulary."""
    with (FIXTURES_DIR / "tiny_vocab.json").open("r", encoding="utf-8") as f:
        return json.load(f)
