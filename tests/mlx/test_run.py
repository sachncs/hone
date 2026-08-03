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
gpu_info = run.gpu_info
has_metal = run.has_metal
read_device = run.read_device
setup_device = run.setup_device


def test_read_device_defaults_to_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HONE_DEVICE", raising=False)
    assert read_device() == "gpu"


def test_read_device_reads_env_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HONE_DEVICE", "CPU")
    assert read_device() == "cpu"
    monkeypatch.setenv("HONE_DEVICE", "GPU")
    assert read_device() == "gpu"


def test_read_device_rejects_unsupported_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HONE_DEVICE", "tpu")
    with pytest.raises(ValueError, match="HONE_DEVICE must be one of"):
        read_device()


def test_setup_device_installs_cpu_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HONE_DEVICE", "cpu")
    original = mlx_core.default_device()
    try:
        setup_device(logging.getLogger("test"))
        assert mlx_core.default_device() == mlx_core.Device(mlx_core.DeviceType.cpu, 0)
    finally:
        mlx_core.set_default_device(original)


def test_setup_device_installs_gpu_when_requested_and_metal_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HONE_DEVICE", "gpu")
    if not has_metal():
        pytest.skip("Metal is not available on this host")
    original = mlx_core.default_device()
    try:
        setup_device(logging.getLogger("test"))
        assert mlx_core.default_device() == mlx_core.Device(mlx_core.DeviceType.gpu, 0)
    finally:
        mlx_core.set_default_device(original)


def test_setup_device_refuses_gpu_when_metal_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HONE_DEVICE", "gpu")
    monkeypatch.setattr(run, "has_metal", lambda: False)
    original = mlx_core.default_device()
    try:
        with pytest.raises(RuntimeError, match="Metal is unavailable"):
            setup_device(logging.getLogger("test"))
    finally:
        mlx_core.set_default_device(original)


def test_setup_logs_device_info() -> None:
    logger = logging.getLogger("test.device_info")
    setup_device(logger)
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
