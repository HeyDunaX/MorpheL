"""Hugging Face Hub upload utilities for MorpheL.

Credentials are always read from the ``HF_TOKEN`` environment variable.
Never pass a token directly as a function argument or CLI flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from morphel.logging_utils import get_logger

logger = get_logger(__name__)


def push_to_hub(
    output_dir: Path,
    repo_id: str,
    token: Optional[str] = None,
) -> None:
    """Upload the complete tokenizer directory to Hugging Face Hub.

    The token is read from the environment variable ``HF_TOKEN`` when not
    provided. Never hard-code or log credentials.

    Args:
        output_dir: Local directory containing all native MorpheL artifacts
            and the Hugging Face wrapper files.
        repo_id: Target Hugging Face repository ID (e.g.
            ``"anonymous-morphel/morphel-tr-32k"``).
        token: Optional Hugging Face API token. If ``None``, the ``HF_TOKEN``
            environment variable is used.

    Raises:
        RuntimeError: If the upload fails.
    """
    import os

    from huggingface_hub import HfApi

    resolved_token = token or os.environ.get("HF_TOKEN")
    if not resolved_token:
        logger.warning(
            "No HF_TOKEN found. Upload will use anonymous access and may fail "
            "for private repositories."
        )

    api = HfApi(token=resolved_token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload MorpheL tokenizer artifacts",
    )
    logger.info("Uploaded tokenizer to https://huggingface.co/%s", repo_id)
