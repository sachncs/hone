"""End-to-end tests for the hone.prepare package.

The prepare layer is the only thing left in ``hone`` after the
Soup-first cut; these tests are the regression guard for the
JSONL files Soup consumes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from hone.prepare import (
    DataError,
    PrepareError,
    PrepareRequest,
    PrepareResult,
    Role,
    prepare_local_file,
)


@pytest.fixture
def tmp_input(tmp_path: Path) -> Path:
    path = tmp_path / "in.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"},
            ]
        }
        for i in range(40)
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def req(tmp_path: Path) -> PrepareRequest:
    return PrepareRequest(
        output=tmp_path / "out",
        seed=42,
        logger=logging.getLogger("test"),
    )


def test_prepare_local_file_splits(tmp_input: Path, req: PrepareRequest) -> None:
    result = prepare_local_file(input_path=tmp_input, request=req, ratio=0.1)
    assert isinstance(result, PrepareResult)
    assert result.written == 40
    assert result.train_count == 36
    assert result.valid_count == 4
    assert (req.output / "train.jsonl").is_file()
    assert (req.output / "valid.jsonl").is_file()


def test_prepare_local_file_drops_empty_rows(
    tmp_path: Path, req: PrepareRequest
) -> None:
    inp = tmp_path / "in.jsonl"
    inp.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "real Q"},
                            {"role": "assistant", "content": "real A"},
                        ]
                    }
                ),
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "  "},
                            {"role": "assistant", "content": "ok"},
                        ]
                    }
                ),
                json.dumps({"prompt": "P", "completion": "C"}),
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(DataError):
        prepare_local_file(input_path=inp, request=req, ratio=0.5)


def test_prepare_local_file_rejects_bad_ratio(
    tmp_input: Path, req: PrepareRequest
) -> None:
    with pytest.raises(DataError, match="ratio"):
        prepare_local_file(input_path=tmp_input, request=req, ratio=0.0)
    with pytest.raises(DataError, match="ratio"):
        prepare_local_file(input_path=tmp_input, request=req, ratio=1.0)


def test_prepare_local_file_rejects_malformed_json(
    tmp_path: Path, req: PrepareRequest
) -> None:
    inp = tmp_path / "in.jsonl"
    inp.write_text('{"messages": []}\nNOT JSON\n', encoding="utf-8")
    with pytest.raises(DataError):
        prepare_local_file(input_path=inp, request=req, ratio=0.5)


def test_prepare_local_file_accepts_prompt_completion(
    tmp_path: Path, req: PrepareRequest
) -> None:
    inp = tmp_path / "in.jsonl"
    inp.write_text(
        "\n".join(
            json.dumps({"prompt": f"Q{i}", "completion": f"A{i}"}) for i in range(3)
        ),
        encoding="utf-8",
    )
    result = prepare_local_file(input_path=inp, request=req, ratio=0.5)
    assert result.written == 3


def test_validation_error_includes_location(
    tmp_input: Path, req: PrepareRequest
) -> None:
    """The location prefix lets callers point at the offending row."""
    with pytest.raises(DataError) as info:
        prepare_local_file(input_path=tmp_input, request=req, ratio=1.5)
    assert "ratio" in str(info.value).lower()


def test_role_enum_values() -> None:
    """Role is exported so downstream code can build chat records."""
    assert Role.user.value == "user"
    assert Role.assistant.value == "assistant"
    assert Role.system.value == "system"


def test_prepare_error_is_base_of_data_error() -> None:
    """All prepare-layer errors share a single catchable base."""
    assert issubclass(DataError, PrepareError)


# ---------------------------------------------------------------------------
# prepare_stream multi-config
# ---------------------------------------------------------------------------


def test_prepare_stream_writes_all_configs(
    tmp_path: Path, req: PrepareRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prepare_stream must drain every config, not just the first.

    Regression: the old `for ... else: continue; break` pattern
    broke the outer loop after a single config. With two synthetic
    iterables both should land in the same all.jsonl.
    """
    from hone.prepare import prepare_stream
    from hone.prepare import service as service_mod

    class _FakeStream:
        def __init__(self, rows: list[dict]) -> None:
            self._rows = rows

        def __iter__(self) -> Iterator[dict[str, object]]:
            return iter(self._rows)

    def _factory(
        repo: str, *, config: str | None = None, split: str = "train"
    ) -> _FakeStream:
        cfg = config or "default"
        return _FakeStream(
            [
                {
                    "messages": [
                        {"role": "user", "content": f"{cfg}-Q"},
                        {"role": "assistant", "content": f"{cfg}-A"},
                    ]
                }
            ]
        )

    monkeypatch.setattr(service_mod, "HubStream", _factory)
    prepare_stream(
        request=req,
        repo="ignored",
        configs=["alpha", "beta"],
        max_tokens=0,
        max_samples=0,
    )

    rows = (req.output / "all.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2
    contents = sorted(json.loads(row)["messages"][0]["content"] for row in rows)
    assert contents == ["alpha-Q", "beta-Q"]


# ---------------------------------------------------------------------------
# Nemotron row normalizer
# ---------------------------------------------------------------------------


def test_nemotron_normalizer_strips_reasoning_content() -> None:
    """The reasoning trace should not leak into the chat template."""
    from hone.prepare.nemotron import _validate_messages

    row: dict[str, object] = {
        "messages": [
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": "```python\nprint()\n```",
                "reasoning_content": "I should use print.",
            },
        ]
    }
    cleaned = _validate_messages(row)
    assert cleaned is not None
    assert cleaned == {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "```python\nprint()\n```"},
        ]
    }


def test_nemotron_normalizer_rejects_empty_messages() -> None:
    from hone.prepare.nemotron import _validate_messages

    assert _validate_messages({"messages": []}) is None
    assert _validate_messages({"messages": [{"role": "user", "content": "  "}]}) is None
    assert _validate_messages({"messages": "not a list"}) is None


def test_nemotron_normalizer_rejects_missing_role() -> None:
    from hone.prepare.nemotron import _validate_messages

    assert (
        _validate_messages(
            {"messages": [{"content": "x"}, {"role": "assistant", "content": "y"}]}
        )
        is None
    )


def test_nemotron_config_is_frozen() -> None:
    """Configs must be immutable so the same name doesn't drift across calls."""
    import dataclasses

    from hone.prepare.nemotron import NEMOTRON_COMPETITIVE_PROGRAMMING

    with pytest.raises(dataclasses.FrozenInstanceError):
        NEMOTRON_COMPETITIVE_PROGRAMMING.name = "something-else"  # type: ignore[misc]
