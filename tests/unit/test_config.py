"""Behavior tests for hone.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from hone.config import REQUIRED_KEYS, load, save, validate
from hone.errors import ConfigError, HoneError


def test_load_reads_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("model: m\ntrain: true\ndata: d\n", encoding="utf-8")
    assert load(path) == {"model": "m", "train": True, "data": "d"}


def test_load_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.yaml")


def test_load_raises_on_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="expected a YAML mapping"):
        load(path)


def test_load_raises_on_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model: m\n  train: true\ndata: d\n", encoding="utf-8")
    with pytest.raises(HoneError):
        load(path)


def test_validate_rejects_missing_model() -> None:
    with pytest.raises(ConfigError, match="model"):
        validate({"train": True, "data": "d"})


def test_validate_rejects_missing_train() -> None:
    with pytest.raises(ConfigError, match="train"):
        validate({"model": "m", "data": "d"})


def test_validate_rejects_missing_data() -> None:
    with pytest.raises(ConfigError, match="data"):
        validate({"model": "m", "train": True})


def test_save_writes_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "out.yaml"
    save(path, {"model": "m", "train": True, "data": "d"})
    assert path.is_file()
    assert load(path) == {"model": "m", "train": True, "data": "d"}


def test_roundtrip_yaml(tmp_path: Path) -> None:
    path = tmp_path / "rt.yaml"
    original = {
        "model": "openbmb/MiniCPM5-1B",
        "train": True,
        "data": "data/processed/code",
        "seed": 42,
        "extra": [1, 2, 3],
    }
    save(path, original)
    assert load(path) == original
    validate(load(path))


def test_required_keys_constant() -> None:
    assert REQUIRED_KEYS == ("model", "train", "data")
