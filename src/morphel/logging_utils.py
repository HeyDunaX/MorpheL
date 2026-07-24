"""Structured logging utilities for MorpheL.

All library code uses these helpers instead of bare ``print()`` statements.
CLI entry points may configure log level and format via ``--log-level``.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a MorpheL module.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)


def configure_root_logger(
    level: str = "INFO",
    fmt: Optional[str] = None,
    stream: Optional[object] = None,
) -> None:
    """Configure the root logger for CLI invocations.

    Args:
        level: Log level string, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
        fmt: Log format string. Defaults to a timestamped format.
        stream: Output stream. Defaults to ``sys.stderr``.
    """
    if fmt is None:
        fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    if stream is None:
        stream = sys.stderr

    handler = logging.StreamHandler(stream)  # type: ignore[arg-type]
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
