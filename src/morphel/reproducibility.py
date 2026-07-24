"""Run reproducibility metadata capture for MorpheL.

Every CLI run records a structured metadata snapshot that allows exact
reproduction of the run environment. The snapshot is written to
``resolved_config.yaml`` in the output directory.
"""

from __future__ import annotations

import datetime
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from morphel.logging_utils import get_logger

logger = get_logger(__name__)


def get_git_commit() -> Optional[str]:
    """Return the current git commit hash, or None if unavailable."""
    try:
        result = subprocess.run(  # noqa: S603 S607
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_package_version(package_name: str) -> Optional[str]:
    """Return the installed version of a package, or None if not installed."""
    try:
        import importlib.metadata

        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_cuda_info() -> dict[str, Optional[str]]:
    """Return CUDA version and GPU name when available."""
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": str(torch.cuda.device_count()),
            }
    except ImportError:
        pass
    return {"cuda_version": None, "gpu_name": None, "gpu_count": None}


def collect_run_metadata(
    config_dict: dict[str, Any],
    seed: int,
    dataset_name: Optional[str] = None,
    dataset_revision: Optional[str] = None,
) -> dict[str, Any]:
    """Collect a reproducibility snapshot for a MorpheL run.

    Args:
        config_dict: The fully resolved configuration as a plain dictionary.
        seed: The random seed used for the run.
        dataset_name: Hugging Face dataset identifier used (optional).
        dataset_revision: Dataset revision/commit hash (optional).

    Returns:
        A dictionary suitable for serialization to ``resolved_config.yaml``.
    """
    import morphel

    cuda_info = get_cuda_info()

    return {
        "morphel_version": morphel.__version__,
        "timestamp_iso8601": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "software": {
            "morphel": morphel.__version__,
            "torch": get_package_version("torch"),
            "transformers": get_package_version("transformers"),
            "tokenizers": get_package_version("tokenizers"),
            "datasets": get_package_version("datasets"),
            "numpy": get_package_version("numpy"),
            "huggingface_hub": get_package_version("huggingface-hub"),
        },
        "hardware": cuda_info,
        "dataset": {
            "name": dataset_name,
            "revision": dataset_revision,
        },
        "resolved_config": config_dict,
    }


def save_run_metadata(
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write the run metadata to ``resolved_config.yaml`` in the output directory.

    Args:
        metadata: Run metadata from :func:`collect_run_metadata`.
        output_dir: Directory where the file will be written.
    """
    import yaml

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "resolved_config.yaml"

    with out_path.open("w", encoding="utf-8") as fh:
        yaml.dump(metadata, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info("Run metadata saved to %s", out_path)
