from __future__ import annotations

from pathlib import Path

import pytest

from musicgen_exp.evaluation import load_jsonl


def test_evaluation_requires_real_probe_results(tmp_path: Path) -> None:
    missing_path = tmp_path / "probe_results.jsonl"

    with pytest.raises(FileNotFoundError, match="probe results JSONL"):
        load_jsonl(missing_path, "probe results JSONL")


def test_evaluation_rejects_empty_real_artifact(tmp_path: Path) -> None:
    empty_path = tmp_path / "probe_results.jsonl"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no rows found"):
        load_jsonl(empty_path, "probe results JSONL")
