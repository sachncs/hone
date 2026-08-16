"""Per-stage subprocess runner with tee'd log capture.

The runner invokes the upstream trainer via the :class:`Launcher`
port, mirrors the child's stdout/stderr to a per-stage log file,
and exposes the captured text so the divergence detector can
scan it after the fact.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Protocol


class Launcher(Protocol):
    """Same port as :class:`hone.backends.Launcher`; re-declared for locality."""

    def run(self, args: list[str], env: dict[str, str]) -> int: ...


class StageRunner:
    """Tee a trainer subprocess to ``log_path``; return code + log text."""

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._captured: str = ""

    @property
    def captured_text(self) -> str:
        """Text captured from the most recent :meth:`run` call."""
        return self._captured

    def run(self, args: list[str], env: dict[str, str]) -> int:
        """Run ``python -m hone.run [...]`` with stdout tee'd to ``log_path``."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "hone.run", *args]
        captured: list[str] = []
        with (
            self._log_path.open("w", encoding="utf-8") as handle,
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            ) as process,
        ):
            stdout = process.stdout
            if stdout is None:
                return_code = process.wait()
                self._captured = ""
                return return_code
            for line in stdout:
                captured.append(line)
                sys.stdout.write(line)
                sys.stdout.flush()
                handle.write(line)
                handle.flush()
            return_code = process.wait()
        self._captured = "".join(captured)
        return return_code


__all__ = ["Launcher", "StageRunner"]
