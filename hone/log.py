"""Application logging configuration."""

from __future__ import annotations

import logging

LOGGER: str = "hone"
FORMAT: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup(verbose: bool = False) -> logging.Logger:
    """Configure and return the package logger.

    Always replaces the StreamHandler so subsequent invocations
    (e.g. across multiple CliRunner.invoke calls) get a fresh
    stream. Multiple handlers on the same logger cause duplicate
    log lines.
    """
    logger = logging.getLogger(LOGGER)
    for sink in list(logger.handlers):
        logger.removeHandler(sink)
    sink = logging.StreamHandler()
    sink.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(sink)
    logger.propagate = False
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


def get(name: str) -> logging.Logger:
    """Return a child logger under the hone package namespace."""
    return logging.getLogger(f"{LOGGER}.{name}")
