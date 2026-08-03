"""Consistent application logging for command-line entry points."""

from __future__ import annotations

import logging

LOGGER_NAME = "finetune"


def configure_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the package logger without using stdout prints."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger
