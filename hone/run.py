"""MLX training launcher entry point.

This is the canonical training entry point for the Apple-Silicon
backend. The CLI and the tune runner invoke
``python -m hone.run --config <path>`` so device selection is
verified and audited in one place.

The module is a thin wrapper around :class:`hone.backends.MlxBackend`;
the heavy lifting (Metal check, device selection, GPU logging)
lives there.
"""

from __future__ import annotations

import sys

from hone.backends import MlxBackend
from hone.log import setup


def main() -> None:
    """Configure the device, then delegate to the MLX LoRA CLI."""
    logger = setup(verbose=False)
    backend = MlxBackend()
    backend.select(logger)
    from mlx_lm.lora import main as mlx_lora_main

    try:
        mlx_lora_main()
    except SystemExit as exit_event:
        if exit_event.code not in (None, 0):
            raise SystemExit(int(exit_event.code)) from exit_event
        return


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:  # pragma: no cover - defensive
        print(f"hone.run: {error}", file=sys.stderr)
        raise SystemExit(1) from error
