"""Behavior tests for hone.train.stages."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hone.train.stages import Stage, full_sequence, select_stages


def test_full_sequence_returns_five_stages_in_order() -> None:
    sequence = full_sequence()
    assert len(sequence) == 5
    repos = [stage.repo for stage in sequence]
    assert repos[0] == "ianncity/KIMI-K2.5-1000000x"
    assert repos[-1] == "microsoft/rStar-Coder"


def test_full_sequence_returns_immutable_tuple() -> None:
    sequence = full_sequence()
    assert isinstance(sequence, tuple)


def test_select_stages_all_returns_full_sequence() -> None:
    sequence = full_sequence()
    assert [s.adapter_path.name for s in select_stages("all")] == [
        s.adapter_path.name for s in sequence
    ]
    assert [s.adapter_path.name for s in select_stages("")] == [
        s.adapter_path.name for s in sequence
    ]


def test_select_stages_by_index_returns_one_stage() -> None:
    selected = select_stages("02")
    assert len(selected) == 1
    assert selected[0].adapter_path.name == "02-codex"


def test_select_stages_by_indices_returns_subset() -> None:
    selected = select_stages("02,03,04")
    expected = ["02-codex", "03-ling", "04-codeforces"]
    assert [s.adapter_path.name for s in selected] == expected


def test_select_stages_by_name_returns_subset() -> None:
    selected = select_stages("02-codex,04-codeforces")
    assert [s.adapter_path.name for s in selected] == ["02-codex", "04-codeforces"]


def test_select_stages_deduplicates_repeated_selectors() -> None:
    selected = select_stages("02,02-codex")
    assert len(selected) == 1


def test_select_stages_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="index out of range"):
        select_stages("99")


def test_select_stages_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="name not found"):
        select_stages("99-notreal")


def test_select_stages_rejects_empty_after_filtering() -> None:
    with pytest.raises(ValueError, match="empty selection"):
        select_stages(",,")


def test_stage_is_frozen_dataclass() -> None:
    stage = Stage(
        repo="r",
        configs="c",
        data_path=Path("d"),
        adapter_path=Path("a"),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        stage.repo = "other"  # type: ignore[misc]
