"""Tests for hone.run and hone.backends: MLX device launcher.

All tests in this module require Apple Silicon Metal and are
guarded by :func:`pytest.importorskip`. On Linux they skip cleanly
without error.
"""

from __future__ import annotations

import importlib
import logging

import pytest

pytestmark = pytest.mark.mlx
mlx_core = pytest.importorskip("mlx.core")

run = importlib.import_module("hone.run")
backends = importlib.import_module("hone.backends")


def _backend(env: str | None = None) -> backends.MlxBackend:
    """Build a backend with an explicit env override (None leaves it)."""
    backend = backends.MlxBackend()
    if env is not None:
        backend._env_var = env  # type: ignore[attr-defined]
    return backend


def test_device_defaults_to_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HONE_DEVICE", raising=False)
    backend = backends.MlxBackend()
    assert backend.device_name == "gpu"


def test_device_reads_env_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = backends.MlxBackend()
    monkeypatch.setenv("HONE_DEVICE", "CPU")
    assert backend.device_name == "cpu"
    monkeypatch.setenv("HONE_DEVICE", "GPU")
    assert backend.device_name == "gpu"


def test_device_rejects_unsupported_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hone.errors import BackendError

    backend = backends.MlxBackend()
    monkeypatch.setenv("HONE_DEVICE", "tpu")
    with pytest.raises(BackendError, match="HONE_DEVICE must be one of"):
        _ = backend.device_name


def test_select_installs_cpu_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = backends.MlxBackend()
    monkeypatch.setenv("HONE_DEVICE", "cpu")
    original = mlx_core.default_device()
    try:
        backend.select(logging.getLogger("test"))
        assert mlx_core.default_device() == mlx_core.Device(mlx_core.DeviceType.cpu, 0)
    finally:
        mlx_core.set_default_device(original)


def test_select_installs_gpu_when_requested_and_metal_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = backends.MlxBackend()
    monkeypatch.setenv("HONE_DEVICE", "gpu")
    if not backend.metal_available():
        pytest.skip("Metal is not available on this host")
    original = mlx_core.default_device()
    try:
        backend.select(logging.getLogger("test"))
        assert mlx_core.default_device() == mlx_core.Device(mlx_core.DeviceType.gpu, 0)
    finally:
        mlx_core.set_default_device(original)


def test_select_refuses_gpu_when_metal_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hone.errors import BackendError

    backend = backends.MlxBackend()
    monkeypatch.setenv("HONE_DEVICE", "gpu")
    monkeypatch.setattr(
        backends.MlxBackend,
        "metal_available",
        staticmethod(lambda: False),
    )
    original = mlx_core.default_device()
    try:
        with pytest.raises(BackendError, match="Metal is unavailable"):
            backend.select(logging.getLogger("test"))
    finally:
        mlx_core.set_default_device(original)


def test_setup_logs_device_info() -> None:
    logger = logging.getLogger("test.device_info")
    backends.MlxBackend().select(logger)
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
    assert frozenset({"cpu", "gpu"}) == backends.DEVICES


def test_scripts_directory_is_gone() -> None:
    """Regression test for T1.1: scripts/ must not exist after the refactor."""
    from pathlib import Path

    assert not (Path("scripts")).exists()
