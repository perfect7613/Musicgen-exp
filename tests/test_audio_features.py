from __future__ import annotations

from pathlib import Path

import pytest

from musicgen_exp.audio_features import load_manifest_jsonl, require_existing_audio


def test_manifest_loader_requires_real_manifest(tmp_path: Path) -> None:
    missing_path = tmp_path / "benchmark_manifest.jsonl"

    with pytest.raises(FileNotFoundError, match="benchmark manifest JSONL"):
        load_manifest_jsonl(missing_path)


def test_audio_loader_requires_real_audio(tmp_path: Path) -> None:
    missing_path = tmp_path / "track.mp3"

    with pytest.raises(FileNotFoundError, match="real benchmark audio"):
        require_existing_audio(missing_path)
