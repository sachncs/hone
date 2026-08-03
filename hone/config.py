"""YAML configuration loader, validator, and writer."""

from __future__ import annotations

from pathlib import Path

import yaml

from hone.types import JsonObject

REQUIRED_KEYS: tuple[str, ...] = ("model", "train", "data")


def load(path: Path) -> JsonObject:
    """Read a YAML config file and return its contents as a dict."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data


def validate(config: JsonObject) -> None:
    """Raise ValueError if any required key is missing from the config."""
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"config missing required keys: {', '.join(missing)}")


def save(path: Path, config: JsonObject) -> None:
    """Write a config dict to a YAML file.

    Creates parent directories if missing. Keys preserve insertion
    order (PyYAML default).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
