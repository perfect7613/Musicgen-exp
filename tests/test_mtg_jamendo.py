from __future__ import annotations

from pathlib import Path

import pytest

from musicgen_exp.mtg_jamendo import (
    audio_path_for_variant,
    build_benchmark_manifest,
    read_audio_licenses,
    read_autotagging_tsv,
)


def test_metadata_reader_requires_real_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "autotagging_instrument.tsv"

    with pytest.raises(FileNotFoundError, match="MTG-Jamendo autotagging metadata TSV"):
        read_autotagging_tsv(missing_path)


def test_license_reader_requires_real_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "audio_licenses.txt"

    with pytest.raises(FileNotFoundError, match="MTG-Jamendo audio_licenses.txt"):
        read_audio_licenses(missing_path)


def test_license_reader_supports_official_block_format(tmp_path: Path) -> None:
    path = tmp_path / "audio_licenses.txt"
    path.write_text(
        "\n".join(
            [
                "00/7400.mp3",
                "Example by Artist from Jamendo: http://www.jamendo.com/track/7400",
                "Available under a Creative Commons license: https://example.com/license",
                "",
            ]
        ),
        encoding="utf-8",
    )

    licenses = read_audio_licenses(path)

    assert licenses["track_0007400"].startswith("Example by Artist")


def test_build_manifest_filters_to_downloaded_audio_low_files(tmp_path: Path) -> None:
    metadata = tmp_path / "autotagging_instrument.tsv"
    metadata.write_text(
        "\n".join(
            [
                "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS",
                "track_0007400\tartist_1\talbum_1\t00/7400.mp3\t120.0\tinstrument---piano",
                "track_0007401\tartist_1\talbum_1\t01/7401.mp3\t120.0\tinstrument---piano",
            ]
        ),
        encoding="utf-8",
    )
    audio_root = tmp_path / "audio"
    audio_root.joinpath("00").mkdir(parents=True)
    audio_root.joinpath("00/7400.low.mp3").write_bytes(b"real placeholder path exists only")

    records = read_autotagging_tsv(metadata)
    manifest = build_benchmark_manifest(
        records=records,
        licenses={"track_0007400": "cc", "track_0007401": "cc"},
        prefix_seconds=10.0,
        continuation_seconds=20.0,
        target_clip_count=1,
        audio_root=audio_root,
        audio_variant="audio-low",
    )

    assert manifest[0].audio_path == "00/7400.low.mp3"


def test_audio_low_variant_rewrites_mp3_path() -> None:
    assert audio_path_for_variant("00/7400.mp3", "audio-low") == "00/7400.low.mp3"
