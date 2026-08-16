"""Backend port (Protocol) and the MLX implementation.

The application layer depends on the two ports (:class:`Backend` for
device selection, :class:`Launcher` for subprocess invocation)
rather than on ``mlx_lm`` or ``mlx.core`` directly. This lets unit
tests stub either independently, and lets future CUDA/Unsloth
backends plug in without touching the orchestration code.

Responsibilities:

* :class:`Backend` — Protocol describing what a fine-tuning backend
  can do (identify itself, select + verify the device).
* :class:`Launcher` — Protocol describing how to invoke a backend
  subprocess with a given argv + env.
* :class:`MlxBackend` — concrete :class:`Backend` for Apple Silicon.
* :class:`SubprocessLauncher` — concrete :class:`Launcher` that runs
  ``python -m hone.run [...]`` in a child process.

The orchestration layers (``hone.cli.train``, ``hone.tune.runner``)
depend on the ports only; concrete classes are wired in by
``hone.cli.__init__``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Protocol

import mlx.core as mx

from hone.errors import BackendError, PipelineError

DEVICES: frozenset[str] = frozenset({"cpu", "gpu"})


class Backend(Protocol):
    """Port that any training backend must satisfy."""

    @property
    def name(self) -> str:
        """Stable identifier for logs / error messages (e.g. ``"mlx"``)."""

    def select(self, logger: logging.Logger) -> None:
        """Pick and verify the active device; log the result."""


class Launcher(Protocol):
    """Port for invoking a training backend subprocess."""

    def run(self, args: list[str], env: dict[str, str]) -> int:
        """Run the upstream trainer with ``args``; return exit code."""


class MlxBackend:
    """MLX backend for Apple Silicon.

    Reads ``HONE_DEVICE``, selects the MLX device, verifies Metal
    when GPU is requested, and emits the active accelerator to the
    log. The class does not invoke training itself; the launcher
    port does that.
    """

    name: str = "mlx"

    def __init__(self, *, env_var: str = "HONE_DEVICE", default: str = "gpu") -> None:
        self._env_var = env_var
        self._default = default

    @property
    def device_name(self) -> str:
        """The currently requested device (``"gpu"`` or ``"cpu"``)."""
        name = os.environ.get(self._env_var, self._default).lower()
        if name not in DEVICES:
            raise BackendError(
                f"{self._env_var} must be one of {sorted(DEVICES)}, got {name!r}"
            )
        return name

    @staticmethod
    def metal_available() -> bool:
        """Return whether MLX can drive a Metal GPU on this host."""
        return bool(getattr(mx.metal, "is_available", lambda: False)())

    @staticmethod
    def gpu_info() -> dict[str, object]:
        """Active GPU device info across MLX versions."""
        getter = getattr(mx, "device_info", None)
        if callable(getter):
            return dict(getter())
        return dict(mx.metal.device_info())

    def select(self, logger: logging.Logger) -> None:
        """Select and verify the MLX device requested by the env var."""
        name = self.device_name
        if name == "gpu":
            if not self.metal_available():
                raise BackendError(
                    f"{self._env_var}=gpu was requested but Metal is unavailable "
                    "on this host. This pipeline targets Apple Silicon; either "
                    "run on a machine with a supported GPU or set "
                    f"{self._env_var}=cpu to fall back to the CPU backend."
                )
            device_type = mx.DeviceType.gpu
            device_label = "Metal GPU"
        else:
            device_type = mx.DeviceType.cpu
            device_label = "CPU"
        mx.set_default_device(mx.Device(device_type, 0))
        logger.info("MLX device: %s (%s=%s)", device_label, self._env_var, name)
        if name == "gpu":
            info = self.gpu_info()
            memory_size = info.get("memory_size", 0)
            logger.info(
                "Metal device: %s, memory=%d bytes, architecture=%s",
                info.get("device_name", "unknown"),
                int(memory_size) if isinstance(memory_size, int) else 0,
                info.get("architecture", "unknown"),
            )


class SubprocessLauncher:
    """Launcher that runs ``python -m hone.run [...]`` in a child process.

    Stderr is forwarded to the caller's stderr so trainer errors are
    not silently swallowed; the trainer's exit code is returned.
    """

    def __init__(self, *, executable: str | None = None) -> None:
        self._executable = executable or sys.executable

    def run(self, args: list[str], env: dict[str, str]) -> int:
        """Invoke the trainer subprocess; return its exit code.

        Raises :class:`PipelineError` only on transport-level
        failures (binary missing, permission denied); the trainer's
        own exit code is returned as-is.
        """
        command = [self._executable, "-m", "hone.run", *args]
        try:
            completed = subprocess.run(
                command,
                env=env,
                check=False,
                text=True,
            )
        except FileNotFoundError as error:
            raise PipelineError(
                f"launcher executable not found: {self._executable}"
            ) from error
        except OSError as error:
            raise PipelineError(
                f"failed to invoke trainer subprocess: {error}"
            ) from error
        return completed.returncode


__all__ = ["Backend", "Launcher", "MlxBackend", "SubprocessLauncher"]
