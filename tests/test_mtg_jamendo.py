from __future__ import annotations

from pathlib import Path

import pytest

from musicgen_exp.mtg_jamendo import read_audio_licenses, read_autotagging_tsv


def test_metadata_reader_requires_real_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "autotagging_instrument.tsv"

    with pytest.raises(FileNotFoundError, match="MTG-Jamendo autotagging metadata TSV"):
        read_autotagging_tsv(missing_path)


def test_license_reader_requires_real_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "audio_licenses.txt"

    with pytest.raises(FileNotFoundError, match="MTG-Jamendo audio_licenses.txt"):
        read_audio_licenses(missing_path)
