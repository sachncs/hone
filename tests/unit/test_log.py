"""Behavior tests for hone.log."""

from __future__ import annotations

import logging

from hone.log import LOGGER, get, setup


def test_setup_returns_logger() -> None:
    logger = setup(verbose=False)
    assert isinstance(logger, logging.Logger)
    assert logger.name == LOGGER


def test_setup_is_idempotent() -> None:
    first = setup(verbose=False)
    handlers_after_first = len(first.handlers)
    second = setup(verbose=True)
    assert first is second
    assert len(second.handlers) == handlers_after_first


def test_setup_respects_verbose_flag() -> None:
    logger = setup(verbose=True)
    assert logger.level == logging.DEBUG
    logger = setup(verbose=False)
    assert logger.level == logging.INFO


def test_get_returns_named_logger() -> None:
    child = get("test")
    assert child.name == f"{LOGGER}.test"
    assert isinstance(child, logging.Logger)
