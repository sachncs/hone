"""Tests for hone.run: MLX device launcher.

All tests in this module require Apple Silicon Metal and are
guarded by pytest.importorskip("mlx.core"). On Linux they
skip cleanly without error.
"""

from __future__ import annotations

import importlib
import logging

import pytest

pytestmark = pytest.mark.mlx
mlx_core = pytest.importorskip("mlx.core")

run = importlib.import_module("hone.run")
DEVICES = run.DEVICES
gpu = run.gpu
metal = run.metal
device = run.device
select = run.select


def test_device_defaults_to_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HONE_DEVICE", raising=False)
    assert device() == "gpu"


def test_device_reads_env_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HONE_DEVICE", "CPU")
    assert device() == "cpu"
    monkeypatch.setenv("HONE_DEVICE", "GPU")
    assert device() == "gpu"


def test_device_rejects_unsupported_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HONE_DEVICE", "tpu")
    with pytest.raises(ValueError, match="HONE_DEVICE must be one of"):
        device()


def test_select_installs_cpu_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HONE_DEVICE", "cpu")
    original = mlx_core.default_device()
    try:
        select(logging.getLogger("test"))
        assert mlx_core.default_device() == mlx_core.Device(mlx_core.DeviceType.cpu, 0)
    finally:
        mlx_core.set_default_device(original)


def test_select_installs_gpu_when_requested_and_metal_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HONE_DEVICE", "gpu")
    if not metal():
        pytest.skip("Metal is not available on this host")
    original = mlx_core.default_device()
    try:
        select(logging.getLogger("test"))
        assert mlx_core.default_device() == mlx_core.Device(mlx_core.DeviceType.gpu, 0)
    finally:
        mlx_core.set_default_device(original)


def test_select_refuses_gpu_when_metal_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HONE_DEVICE", "gpu")
    monkeypatch.setattr(run, "metal", lambda: False)
    original = mlx_core.default_device()
    try:
        with pytest.raises(RuntimeError, match="Metal is unavailable"):
            select(logging.getLogger("test"))
    finally:
        mlx_core.set_default_device(original)


def test_setup_logs_device_info() -> None:
    logger = logging.getLogger("test.device_info")
    select(logger)
    assert logger.level <= logging.INFO


def test_run_runs_as_module() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "hone.run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--config" in completed.stdout


def test_run_help_works() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "hone.run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0


def test_devices_constant() -> None:
    assert frozenset({"cpu", "gpu"}) == DEVICES


def test_scripts_directory_is_gone() -> None:
    """Regression test for T1.1: scripts/ must not exist after the refactor."""
    from pathlib import Path

    assert not (Path("scripts")).exists()
