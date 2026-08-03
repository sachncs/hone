"""Tests for the explicit MLX device-selection launcher."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import mlx.core as mx
import pytest

run_mlx_lora = importlib.import_module("scripts.run_mlx_lora")
configure_device = run_mlx_lora.configure_device
metal_available = run_mlx_lora.metal_available
requested_device = run_mlx_lora.requested_device


def test_requested_device_defaults_to_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FINETUNE_DEVICE", raising=False)

    assert requested_device() == "gpu"


def test_requested_device_reads_env_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINETUNE_DEVICE", "CPU")
    assert requested_device() == "cpu"
    monkeypatch.setenv("FINETUNE_DEVICE", "GPU")
    assert requested_device() == "gpu"


def test_requested_device_rejects_unsupported_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINETUNE_DEVICE", "tpu")

    with pytest.raises(ValueError, match="FINETUNE_DEVICE must be one of"):
        requested_device()


def test_configure_device_installs_cpu_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINETUNE_DEVICE", "cpu")
    original = mx.default_device()

    try:
        configure_device(logging.getLogger("test"))
        assert mx.default_device() == mx.Device(mx.DeviceType.cpu, 0)
    finally:
        mx.set_default_device(original)


def test_configure_device_installs_gpu_when_requested_and_metal_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINETUNE_DEVICE", "gpu")
    if not metal_available():
        pytest.skip("Metal is not available on this host")
    original = mx.default_device()

    try:
        configure_device(logging.getLogger("test"))
        assert mx.default_device() == mx.Device(mx.DeviceType.gpu, 0)
    finally:
        mx.set_default_device(original)


def test_configure_device_refuses_gpu_when_metal_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINETUNE_DEVICE", "gpu")
    monkeypatch.setattr(run_mlx_lora, "metal_available", lambda: False)
    original = mx.default_device()

    try:
        with pytest.raises(RuntimeError, match="Metal is unavailable"):
            configure_device(logging.getLogger("test"))
    finally:
        mx.set_default_device(original)


@pytest.mark.parametrize(
    "script_name",
    ["train_stage.sh", "train_full_sequence.sh", "train_mlx.sh"],
)
def test_shell_script_invokes_device_launcher(
    script_name: str,
) -> None:
    script_path = Path("scripts") / script_name
    contents = script_path.read_text(encoding="utf-8")

    assert "run_mlx_lora.py" in contents, (
        f"{script_name} must invoke scripts/run_mlx_lora.py so device "
        f"selection is explicit and verified"
    )
    assert "mlx_lm.lora" not in contents, (
        f"{script_name} must not bypass scripts/run_mlx_lora.py and call "
        f"mlx_lm.lora directly"
    )
