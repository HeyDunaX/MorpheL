"""Significance report generation.

Exports the results of statistical tests as machine-readable JSON, raw CSV for
aggregators, and Markdown for human inspection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_significance_report(
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write significance results to disk.

    Writes:
    - significance_report.json
    - significance_summary.md

    Args:
        results: Dictionary containing all test outputs and metadata.
        output_dir: Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "significance_report.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    with (output_dir / "significance_summary.md").open("w", encoding="utf-8") as fh:
        fh.write("# Statistical Significance Report\n\nStub: implement markdown export.\n")
