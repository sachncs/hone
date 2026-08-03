"""MLX device launcher entry point.

Reads HONE_DEVICE, selects the MLX backend explicitly, verifies
Metal is available when GPU is requested, logs the active
accelerator, then delegates to the upstream mlx_lm.lora CLI.

This is the canonical training launcher for the Apple-Silicon
backend. Every training entry point (hone train, hone tune)
invokes 'python -m hone.run --config <path>' so device selection
is verified and audited in one place.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx

from hone.log import setup

DEVICES: frozenset[str] = frozenset({"cpu", "gpu"})


def read_device() -> str:
    """Return the device name requested by HONE_DEVICE."""
    device_name = os.environ.get("HONE_DEVICE", "gpu").lower()
    if device_name not in DEVICES:
        raise ValueError(
            f"HONE_DEVICE must be one of {sorted(DEVICES)}, got {device_name!r}"
        )
    return device_name


def has_metal() -> bool:
    """Return whether MLX can drive a Metal GPU on this host."""
    return bool(getattr(mx.metal, "is_available", lambda: False)())


def gpu_info() -> dict[str, Any]:
    """Return the active GPU device info across MLX versions.

    Newer MLX exposes mx.device_info; older releases only expose
    mx.metal.device_info. Use whichever is available.
    """
    getter = getattr(mx, "device_info", None)
    if callable(getter):
        return dict(getter())
    return dict(mx.metal.device_info())


def setup_device(logger: logging.Logger) -> None:
    """Select and verify the MLX device requested by HONE_DEVICE.

    The selected device is installed as the MLX default and the
    Metal device identity is logged so the active accelerator is
    auditable from the training log. A clear error is raised when
    GPU is requested but Metal is unavailable, so the pipeline
    never silently degrades to CPU.
    """
    device_name = read_device()
    if device_name == "gpu":
        if not has_metal():
            raise RuntimeError(
                "HONE_DEVICE=gpu was requested but Metal is unavailable "
                "on this host. This pipeline targets Apple Silicon; either "
                "run on a machine with a supported GPU or set "
                "HONE_DEVICE=cpu to fall back to the CPU backend."
            )
        device_type = mx.DeviceType.gpu
        device_label = "Metal GPU"
    else:
        device_type = mx.DeviceType.cpu
        device_label = "CPU"
    mx.set_default_device(mx.Device(device_type, 0))
    logger.info("MLX device: %s (HONE_DEVICE=%s)", device_label, device_name)
    if device_name == "gpu":
        info = gpu_info()
        memory_size = info.get("memory_size", 0)
        logger.info(
            "Metal device: %s, memory=%d bytes, architecture=%s",
            info.get("device_name", "unknown"),
            int(memory_size) if isinstance(memory_size, int) else 0,
            info.get("architecture", "unknown"),
        )


def main() -> None:
    """Configure the device, then delegate to the MLX LoRA CLI."""
    logger = setup(verbose=False)
    setup_device(logger)
    from mlx_lm.lora import main as mlx_lora_main

    mlx_lora_main()


if __name__ == "__main__":
    main()
