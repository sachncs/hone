"""Integration tests for hone.cli.train: dispatch verification.

These tests assert that the CLI correctly builds the subprocess
invocation, sets HONE_DEVICE in the environment, and surfaces the
subprocess exit code. The actual training step is mocked so the
tests do not depend on the mlx_lm trainer behavior on CPU devices.

`train code` / `train swe` invoke the trainer via subprocess.run;
`train all` uses a tee wrapper (_run_stage_capture) so the watchdog
can scan the captured output, and that is the seam these tests patch.

Full training runs are exercised in tests/mlx/test_run.py on
Apple Silicon with a real model.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click import unstyle
from typer.testing import CliRunner

from hone.cli import app
from hone.cli import train as train_module

pytestmark = pytest.mark.mlx

runner = CliRunner()


def test_train_code_refuses_nonexistent_config(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "train",
            "code",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code != 0
    assert "config not found" in result.output or "config not found" in (
        result.stderr or ""
    )


def test_train_swe_refuses_nonexistent_config(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "train",
            "swe",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code != 0
    assert "config not found" in result.output or "config not found" in (
        result.stderr or ""
    )


def test_train_code_refuses_unknown_backend(tmp_path: Path) -> None:
    config = tmp_path / "code.yaml"
    config.write_text("model: m\ntrain: true\ndata: d\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["train", "code", "--config", str(config), "--backend", "tpu"],
    )
    assert result.exit_code != 0
    assert "backend" in result.output.lower()


@pytest.fixture
def captured_subprocess() -> dict[str, object]:
    return {}


@pytest.fixture
def captured_stage() -> dict[str, object]:
    return {}


def make_capture(
    target: dict[str, object],
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def capture(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Fake subprocess.run that records args and env into `target`."""
        command: Sequence[str] = kwargs.get("args", args[0] if args else [])
        env: dict[str, str] = kwargs.get("env", {})
        target["command"] = command
        target["env"] = env
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return capture


def stage_capture(
    target: dict[str, object],
    log_text: str = "",
    return_code: int = 0,
) -> Callable[..., tuple[int, str]]:
    def fake(cmd: list[str], env: dict[str, str], log_path: Path) -> tuple[int, str]:
        target["command"] = list(cmd)
        target["env"] = dict(env)
        target["log_path"] = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(log_text, encoding="utf-8")
        return return_code, log_text

    return fake


def test_train_code_sets_hone_device_and_invokes_subprocess(
    tmp_path: Path, captured_subprocess: dict[str, object]
) -> None:
    config = tmp_path / "code.yaml"
    config.write_text("model: m\ntrain: true\ndata: d\n", encoding="utf-8")
    with patch.object(
        train_module.subprocess, "run", side_effect=make_capture(captured_subprocess)
    ):
        result = runner.invoke(
            app,
            ["train", "code", "--config", str(config), "--device", "cpu"],
        )
    assert result.exit_code == 0
    command = captured_subprocess["command"]
    env = captured_subprocess["env"]
    assert isinstance(command, list)
    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "hone.run"]
    assert "--config" in command
    assert str(config) in command
    assert isinstance(env, dict)
    assert env.get("HONE_DEVICE") == "cpu"


def test_train_swe_sets_hone_device_and_invokes_subprocess(
    tmp_path: Path, captured_subprocess: dict[str, object]
) -> None:
    config = tmp_path / "swe.yaml"
    config.write_text("model: m\ntrain: true\ndata: d\n", encoding="utf-8")
    with patch.object(
        train_module.subprocess, "run", side_effect=make_capture(captured_subprocess)
    ):
        result = runner.invoke(
            app,
            ["train", "swe", "--config", str(config), "--device", "gpu"],
        )
    assert result.exit_code == 0
    env = captured_subprocess["env"]
    assert isinstance(env, dict)
    assert env.get("HONE_DEVICE") == "gpu"


def test_full_sequence_has_expected_datasets() -> None:
    """The full sequence must include all 5 documented datasets in order."""
    from hone.cli.train import FULL_SEQUENCE

    repos = [entry[0] for entry in FULL_SEQUENCE]
    assert repos == [
        "ianncity/KIMI-K2.5-1000000x",
        "Modotte/CodeX-7M-Non-Thinking",
        "inclusionAI/Ling-Coder-SFT",
        "open-r1/codeforces",
        "microsoft/rStar-Coder",
    ]
    adapters = [entry[3] for entry in FULL_SEQUENCE]
    assert adapters == [
        Path("artifacts/full/01-kimi"),
        Path("artifacts/full/02-codex"),
        Path("artifacts/full/03-ling"),
        Path("artifacts/full/04-codeforces"),
        Path("artifacts/full/05-rstar"),
    ]


def test_full_sequence_adapter_directories_are_unique() -> None:
    """Each stage must write to a distinct adapter directory."""
    from hone.cli.train import FULL_SEQUENCE

    adapters = [entry[3] for entry in FULL_SEQUENCE]
    assert len(set(adapters)) == len(adapters)


def test_train_all_help_runs() -> None:
    """The all subcommand must be registered and parseable."""
    result = runner.invoke(app, ["train", "all", "--help"])
    assert result.exit_code == 0
    assert "--model" in unstyle(result.stdout)


def chat_jsonl(path: Path, count: int) -> None:
    """Write count distinct chat records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(count):
            record = {
                "messages": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ]
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def value_after(command: object, flag: str) -> str:
    assert isinstance(command, list)
    assert flag in command
    return str(command[command.index(flag) + 1])


def test_train_all_creates_valid_split_before_training(
    tmp_path: Path, captured_stage: dict[str, object]
) -> None:
    train_path = tmp_path / "data" / "01-stage" / "train.jsonl"
    chat_jsonl(train_path, 40)
    train_module._stage_marker(train_path, 4096, 0).write_text(
        json.dumps({"max_tokens": 4096}), encoding="utf-8"
    )
    fake_sequence = [
        ("fake/repo", "default", train_path, tmp_path / "adapters" / "01-stage")
    ]
    with (
        patch.object(train_module, "FULL_SEQUENCE", fake_sequence),
        patch.object(
            train_module,
            "_run_stage_capture",
            side_effect=stage_capture(captured_stage),
        ),
    ):
        result = runner.invoke(app, ["train", "all", "--model", "m"])
    assert result.exit_code == 0

    valid_path = train_path.parent / "valid.jsonl"
    assert valid_path.is_file()
    assert valid_path.stat().st_size > 0
    train_lines = len(train_path.read_text(encoding="utf-8").splitlines())
    valid_lines = len(valid_path.read_text(encoding="utf-8").splitlines())
    assert train_lines + valid_lines == 40

    command = captured_stage["command"]
    assert value_after(command, "--data") == str(train_path.parent)
    assert value_after(command, "--iters") == str(train_lines)


def test_train_all_skips_split_when_valid_present(
    tmp_path: Path, captured_stage: dict[str, object]
) -> None:
    train_path = tmp_path / "data" / "01-stage" / "train.jsonl"
    chat_jsonl(train_path, 40)
    train_module._stage_marker(train_path, 4096, 0).write_text(
        json.dumps({"max_tokens": 4096}), encoding="utf-8"
    )
    valid_path = train_path.parent / "valid.jsonl"
    chat_jsonl(valid_path, 2)
    valid_before = valid_path.read_text(encoding="utf-8")

    fake_sequence = [
        ("fake/repo", "default", train_path, tmp_path / "adapters" / "01-stage")
    ]
    with (
        patch.object(train_module, "FULL_SEQUENCE", fake_sequence),
        patch.object(
            train_module,
            "_run_stage_capture",
            side_effect=stage_capture(captured_stage),
        ),
    ):
        result = runner.invoke(app, ["train", "all", "--model", "m"])
    assert result.exit_code == 0

    assert valid_path.read_text(encoding="utf-8") == valid_before
    command = captured_stage["command"]
    assert value_after(command, "--iters") == "40"


def test_train_all_writes_prepare_marker_after_prepare(
    tmp_path: Path, captured_stage: dict[str, object]
) -> None:
    """When the JSONL is freshly built by prepare, a marker is written."""
    train_path = tmp_path / "data" / "01-stage" / "train.jsonl"
    chat_jsonl(train_path, 40)

    from hone.cli import prepare as prepare_module

    def fake_prepare(*args: Any, **kwargs: Any) -> None:
        train_path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"q{i}"},
                            {"role": "assistant", "content": f"a{i}"},
                        ]
                    }
                )
                for i in range(40)
            )
            + "\n",
            encoding="utf-8",
        )

    with (
        patch.object(
            train_module,
            "FULL_SEQUENCE",
            [("fake/repo", "default", train_path, tmp_path / "adapters" / "01-stage")],
        ),
        patch.object(prepare_module, "all_cmd", side_effect=fake_prepare),
        patch.object(
            train_module,
            "_run_stage_capture",
            side_effect=stage_capture(captured_stage),
        ),
    ):
        result = runner.invoke(
            app, ["train", "all", "--model", "m", "--max-tokens", "4096"]
        )
    assert result.exit_code == 0
    marker = train_module._stage_marker(train_path, 4096, 0)
    assert marker.is_file()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["max_tokens"] == 4096


def test_train_all_rebuilds_when_marker_for_different_max_tokens(
    tmp_path: Path, captured_stage: dict[str, object]
) -> None:
    """A re-run with a different --max-tokens must trigger re-prepare."""
    train_path = tmp_path / "data" / "01-stage" / "train.jsonl"
    chat_jsonl(train_path, 40)
    train_module._stage_marker(train_path, 2048, 0).write_text(
        json.dumps({"max_tokens": 2048}), encoding="utf-8"
    )

    from hone.cli import prepare as prepare_module

    prepare_calls: list[dict[str, Any]] = []

    def fake_prepare(*args: Any, **kwargs: Any) -> None:
        prepare_calls.append(kwargs)
        chat_jsonl(train_path, 40)

    with (
        patch.object(
            train_module,
            "FULL_SEQUENCE",
            [("fake/repo", "default", train_path, tmp_path / "adapters" / "01-stage")],
        ),
        patch.object(prepare_module, "all_cmd", side_effect=fake_prepare),
        patch.object(
            train_module,
            "_run_stage_capture",
            side_effect=stage_capture(captured_stage),
        ),
    ):
        result = runner.invoke(
            app, ["train", "all", "--model", "m", "--max-tokens", "4096"]
        )
    assert result.exit_code == 0
    assert len(prepare_calls) == 1
    assert prepare_calls[0]["max_tokens"] == 4096
    marker = train_module._stage_marker(train_path, 4096, 0)
    assert marker.is_file()


def test_train_all_aborts_on_nan_losses_and_removes_adapter(
    tmp_path: Path, captured_stage: dict[str, object]
) -> None:
    """A divergent stage produces NaN losses → run aborts, adapter deleted."""
    train_path = tmp_path / "data" / "01-stage" / "train.jsonl"
    chat_jsonl(train_path, 40)
    train_module._stage_marker(train_path, 4096, 0).write_text(
        json.dumps({"max_tokens": 4096}), encoding="utf-8"
    )
    adapter_dir = tmp_path / "adapters" / "01-stage"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapters.safetensors").write_text("weights", encoding="utf-8")
    nan_log = "\n".join(f"Iter {i}: Train loss nan" for i in (10, 20, 30, 40)) + "\n"

    fake_sequence = [("fake/repo", "default", train_path, adapter_dir)]
    with (
        patch.object(train_module, "FULL_SEQUENCE", fake_sequence),
        patch.object(
            train_module,
            "_run_stage_capture",
            side_effect=stage_capture(captured_stage, log_text=nan_log),
        ),
    ):
        result = runner.invoke(app, ["train", "all", "--model", "m"])
    assert result.exit_code == 2
    assert not adapter_dir.exists()


def test_train_all_selects_stages_by_index(tmp_path: Path) -> None:
    selected = train_module._select_stages("02")
    assert len(selected) == 1
    assert selected[0][3].name == "02-codex"


def test_train_all_selects_code_only_by_name(tmp_path: Path) -> None:
    selected = train_module._select_stages("02-codex,03-ling,04-codeforces,05-rstar")
    assert [entry[3].name for entry in selected] == [
        "02-codex",
        "03-ling",
        "04-codeforces",
        "05-rstar",
    ]
    repos = [entry[0] for entry in selected]
    assert "ianncity/KIMI-K2.5-1000000x" not in repos


def test_train_all_select_stages_rejects_unknown_name() -> None:
    result = runner.invoke(app, ["train", "all", "--stages", "99-notreal"])
    assert result.exit_code != 0
    assert "name not found" in (result.output + (result.stderr or ""))


def test_train_all_rejects_max_tokens_greater_than_seq_len() -> None:
    result = runner.invoke(
        app,
        ["train", "all", "--model", "m", "--max-tokens", "8192", "--seq-len", "4096"],
    )
    assert result.exit_code != 0


def test_is_divergent_threshold() -> None:
    few_nans = "Iter 10: Train loss nan\nIter 20: Train loss 1.5\n"
    many_nans = "\n".join(f"Iter {i}: Train loss nan" for i in (10, 20, 30))
    assert not train_module._is_divergent(few_nans)
    assert train_module._is_divergent(many_nans)


def test_stage_marker_keyed_on_max_samples(tmp_path: Path) -> None:
    """Different --max-samples values produce different markers; both stored."""
    train_path = tmp_path / "data" / "01-stage" / "train.jsonl"
    marker_4096 = train_module._stage_marker(train_path, 4096, 0)
    marker_4096_500 = train_module._stage_marker(train_path, 4096, 500)
    marker_2048 = train_module._stage_marker(train_path, 2048, 0)
    assert marker_4096 != marker_4096_500
    assert marker_4096 != marker_2048
    assert "samples-500" in marker_4096_500.name
    assert "samples-0" in marker_4096.name

