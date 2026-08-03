"""Application logging configuration."""

from __future__ import annotations

import logging

LOGGER: str = "hone"
_FORMAT: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup(verbose: bool = False) -> logging.Logger:
    """Configure and return the package logger.

    Idempotent: calling setup() twice does not duplicate handlers.
    verbose=True sets DEBUG level; verbose=False sets INFO.
    """
    logger = logging.getLogger(LOGGER)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


def get(name: str) -> logging.Logger:
    """Return a child logger under the hone package namespace."""
    return logging.getLogger(f"{LOGGER}.{name}")
