from __future__ import annotations

from pathlib import Path

import pytest

from musicgen_exp.probes import load_activation_index, load_annotation_splits


def test_probe_runner_requires_real_activation_index(tmp_path: Path) -> None:
    missing_path = tmp_path / "activation_index.jsonl"

    with pytest.raises(FileNotFoundError, match="activation index JSONL"):
        load_activation_index(missing_path)


def test_probe_runner_requires_real_annotation_splits(tmp_path: Path) -> None:
    missing_path = tmp_path / "annotation_splits.json"

    with pytest.raises(FileNotFoundError, match="annotation splits JSON"):
        load_annotation_splits(missing_path)
