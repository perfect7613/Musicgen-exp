from __future__ import annotations

from pathlib import Path

import pytest

from musicgen_exp.annotation_workflow import assign_split, load_annotation_jsonl


def test_annotation_loader_requires_real_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "verified.jsonl"

    with pytest.raises(FileNotFoundError, match="annotation JSONL"):
        load_annotation_jsonl(missing_path)


def test_split_assignment_is_deterministic_for_track_id() -> None:
    first = assign_split("1376256", train_ratio=0.7, validation_ratio=0.15)
    second = assign_split("1376256", train_ratio=0.7, validation_ratio=0.15)

    assert first == second
    assert first in {"train", "validation", "test"}


def test_invalid_split_ratios_are_rejected() -> None:
    with pytest.raises(ValueError, match="split ratios"):
        assign_split("1376256", train_ratio=0.9, validation_ratio=0.2)
