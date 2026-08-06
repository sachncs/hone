"""Behavior tests for hone.jsonl: Reader and Writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from hone.jsonl import Reader, Writer
from hone.model import Example, Message, Role


def examples(count: int) -> list[Example]:
    return [
        Example(
            messages=(
                Message(role=Role.user, content=f"q{i}"),
                Message(role=Role.assistant, content=f"a{i}"),
            ),
            metadata={"row_id": i, "score": float(i)},
        )
        for i in range(count)
    ]


def test_writer_roundtrips_messages(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    written = examples(3)
    Writer().write(path, written)
    loaded = list(Reader().read(path))
    assert loaded == written


def test_writer_roundtrips_metadata(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    examples = [
        Example(
            messages=(
                Message(role=Role.user, content="q"),
                Message(role=Role.assistant, content="a"),
            ),
            metadata={"k": 1, "s": "x", "f": 1.5, "b": True, "n": None},
        )
    ]
    Writer().write(path, examples)
    loaded = list(Reader().read(path))
    assert loaded[0].metadata == {"k": 1, "s": "x", "f": 1.5, "b": True, "n": None}


def test_writer_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "train.jsonl"
    Writer().write(path, examples(2))
    assert path.is_file()


def test_writer_uses_utf8(tmp_path: Path) -> None:
    path = tmp_path / "u.jsonl"
    example = Example(
        messages=(
            Message(role=Role.user, content="日本語"),
            Message(role=Role.assistant, content="中文"),
        )
    )
    Writer().write(path, [example])
    text = path.read_text(encoding="utf-8")
    assert "日本語" in text
    assert "中文" in text


def test_writer_serializes_unicode(tmp_path: Path) -> None:
    path = tmp_path / "u.jsonl"
    example = Example(
        messages=(
            Message(role=Role.user, content="emoji 🎉"),
            Message(role=Role.assistant, content="✓"),
        )
    )
    Writer().write(path, [example])
    text = path.read_text(encoding="utf-8")
    assert "\\u" not in text
    assert "🎉" in text


def test_writer_returns_count(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    count = Writer().write(path, examples(5))
    assert count == 5


def test_reader_reports_line_number_for_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"messages": [{"role": "user", "content": "q"}, '
        '{"role": "assistant", "content": "a"}]}\n'
        "NOT JSON\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r":2:"):
        list(Reader().read(path))


def test_reader_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "blank.jsonl"
    expected = Example(
        messages=(
            Message(role=Role.user, content="q"),
            Message(role=Role.assistant, content="a"),
        ),
        metadata={"row_id": 0, "score": 0.0},
    )
    path.write_text(
        "\n" + '{"messages": [{"role": "user", "content": "q"}, '
        '{"role": "assistant", "content": "a"}], "row_id": 0, "score": 0.0}\n' + "\n\n",
        encoding="utf-8",
    )
    loaded = list(Reader().read(path))
    assert len(loaded) == 1
    assert loaded[0] == expected


def test_reader_rejects_non_dict_record(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":1: each JSONL record must be an object"):
        list(Reader().read(path))


def test_reader_rejects_missing_messages(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"foo": "bar"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"record must contain a 'messages' list"):
        list(Reader().read(path))


def test_reader_rejects_non_list_messages(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"messages": "not a list"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"record must contain a 'messages' list"):
        list(Reader().read(path))


def test_reader_rejects_non_object_message_items(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"messages": ["bad"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"messages\[0\] must be an object"):
        list(Reader().read(path))
