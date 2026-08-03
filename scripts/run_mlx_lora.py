#!/usr/bin/env python3
"""Launch MLX LoRA with an explicit Metal GPU or CPU device selection.

This is the canonical training launcher for the Apple-Silicon backend. It
selects the MLX device before any training code runs and fails fast if the
requested device is unavailable, so every shell script invokes the trainer
through the same verified entry point instead of relying on MLX's implicit
default.
"""

from __future__ import annotations

import logging
import os
import sys

import mlx.core as mx

from finetune.logging import configure_logging

SUPPORTED_DEVICES: frozenset[str] = frozenset({"cpu", "gpu"})


def requested_device() -> str:
    """Return the device name requested by ``FINETUNE_DEVICE``."""
    device_name = os.environ.get("FINETUNE_DEVICE", "gpu").lower()
    if device_name not in SUPPORTED_DEVICES:
        raise ValueError(
            f"FINETUNE_DEVICE must be one of {sorted(SUPPORTED_DEVICES)}, "
            f"got {device_name!r}"
        )
    return device_name


def metal_available() -> bool:
    """Return whether MLX can drive a Metal GPU on this host."""
    return bool(getattr(mx.metal, "is_available", lambda: False)())


def gpu_device_info() -> dict[str, object]:
    """Return the active GPU device info across MLX versions.

    Newer MLX exposes ``mx.device_info``; older releases only expose
    ``mx.metal.device_info``. Use whichever is available so the launcher
    reports the active accelerator on every supported MLX release.
    """
    getter = getattr(mx, "device_info", None)
    if callable(getter):
        return dict(getter())
    return dict(mx.metal.device_info())


def configure_device(logger: logging.Logger) -> None:
    """Select and verify the MLX device requested by ``FINETUNE_DEVICE``.

    The selected device is installed as the MLX default and the Metal device
    identity is logged so the active accelerator is auditable from the
    training log. A clear error is raised when GPU is requested but Metal
    is unavailable, so the pipeline never silently degrades to CPU.
    """
    device_name = requested_device()
    if device_name == "gpu":
        if not metal_available():
            raise RuntimeError(
                "FINETUNE_DEVICE=gpu was requested but Metal is unavailable "
                "on this host. This pipeline targets Apple Silicon; either "
                "run on a machine with a supported GPU or set "
                "FINETUNE_DEVICE=cpu to fall back to the CPU backend."
            )
        device_type = mx.DeviceType.gpu
        device_label = "Metal GPU"
    else:
        device_type = mx.DeviceType.cpu
        device_label = "CPU"
    mx.set_default_device(mx.Device(device_type, 0))
    logger.info("MLX device: %s (FINETUNE_DEVICE=%s)", device_label, device_name)
    if device_name == "gpu":
        info = gpu_device_info()
        memory_size = info.get("memory_size", 0)
        logger.info(
            "Metal device: %s, memory=%d bytes, architecture=%s",
            info.get("device_name", "unknown"),
            int(memory_size) if isinstance(memory_size, int) else 0,
            info.get("architecture", "unknown"),
        )


def main() -> None:
    """Configure the device, then delegate to the MLX LoRA CLI."""
    logger = configure_logging(verbose=False)
    configure_device(logger)
    from mlx_lm.lora import main as run_lora

    sys.exit(run_lora())


if __name__ == "__main__":
    main()
