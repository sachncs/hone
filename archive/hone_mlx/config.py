"""YAML configuration loader, validator, and writer.

Thin wrapper over PyYAML — kept here so future config-schema work
(schemas, defaults, env-var resolution) has a single place to live.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hone.errors import ConfigError

REQUIRED_KEYS: tuple[str, ...] = ("model", "train", "data")


def load(path: Path) -> dict[str, Any]:
    """Read a YAML config file and return its contents as a dict."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data


def validate(config: dict[str, Any]) -> None:
    """Raise :class:`ConfigError` if any required key is missing."""
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ConfigError(f"config missing required keys: {', '.join(missing)}")


def save(path: Path, config: dict[str, Any]) -> None:
    """Write a config dict to a YAML file.

    Creates parent directories if missing. Keys preserve insertion
    order (PyYAML default).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


__all__ = ["REQUIRED_KEYS", "load", "save", "validate"]
