"""Domain exception hierarchy.

All errors raised by hone's domain, application, and infrastructure
layers are subclasses of ``HoneError``. The CLI transport layer
catches the entire hierarchy and converts to ``typer.BadParameter``,
so callers never see a generic ``ValueError`` or ``RuntimeError``
when something hone-specific fails.

The hierarchy is deliberately shallow:

* ``HoneError`` — base, carries a message and optional cause.
* ``ConfigError`` — invalid or missing configuration.
* ``DataError`` — malformed, missing, or unparseable dataset.
* ``ValidationError`` — record-level rejection (subset of DataError).
* ``BackendError`` — training/inference backend failure.
* ``PipelineError`` — orchestration-level failure (subprocess, missing
  artifact, sidecar mismatch).

Keeping the hierarchy shallow makes the catch in the CLI layer
explicit (one ``except HoneError``) and avoids a deep ``except``
ladder downstream.
"""

from __future__ import annotations


class HoneError(Exception):
    """Base class for every hone-raised exception."""


class ConfigError(HoneError):
    """Configuration is invalid, missing, or unreadable."""


class DataError(HoneError):
    """Dataset is malformed, missing, or unreadable."""


class ValidationError(DataError):
    """A record failed validation; carries the offending location."""


class BackendError(HoneError):
    """A training or inference backend raised a non-recoverable error."""


class PipelineError(HoneError):
    """Multi-stage orchestration failed (subprocess, missing artifact)."""


__all__ = [
    "BackendError",
    "ConfigError",
    "DataError",
    "HoneError",
    "PipelineError",
    "ValidationError",
]
