from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable


DATASET_NAME = "MTG-Jamendo"


@dataclass(frozen=True)
class TrackRecord:
    track_id: str
    artist_id: str
    album_id: str
    audio_path: str
    duration_seconds: float
    tags: list[str]

    @property
    def instrument_tags(self) -> list[str]:
        return sorted(tag.removeprefix("instrument---") for tag in self.tags if tag.startswith("instrument---"))


@dataclass(frozen=True)
class BenchmarkManifestItem:
    track_id: str
    source_dataset: str
    license: str
    source_url: str
    audio_path: str
    duration_seconds: float
    prefix_window: dict[str, float]
    generation_window: dict[str, float]
    tags: list[str]
    instrument_tags: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def read_autotagging_tsv(path: str | Path) -> list[TrackRecord]:
    metadata_path = require_existing_file(path, "MTG-Jamendo autotagging metadata TSV")
    records: list[TrackRecord] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                raise ValueError(
                    f"{metadata_path}:{line_number}: expected at least 6 tab-separated fields"
                )
            track_id, artist_id, album_id, audio_path, duration_raw, *tags = parts
            try:
                duration_seconds = float(duration_raw)
            except ValueError as exc:
                raise ValueError(
                    f"{metadata_path}:{line_number}: invalid duration {duration_raw!r}"
                ) from exc
            records.append(
                TrackRecord(
                    track_id=track_id,
                    artist_id=artist_id,
                    album_id=album_id,
                    audio_path=audio_path,
                    duration_seconds=duration_seconds,
                    tags=tags,
                )
            )
    return records


def read_audio_licenses(path: str | Path) -> dict[str, str]:
    licenses_path = require_existing_file(path, "MTG-Jamendo audio_licenses.txt")
    licenses: dict[str, str] = {}
    with licenses_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"{licenses_path}:{line_number}: expected track id and license fields"
                )
            track_id, license_value = parts[0], parts[-1]
            licenses[track_id] = license_value
    return licenses


def build_benchmark_manifest(
    records: Iterable[TrackRecord],
    licenses: dict[str, str],
    prefix_seconds: float,
    continuation_seconds: float,
    target_clip_count: int,
    excluded_tag_fragments: Iterable[str] = ("voice", "vocal"),
) -> list[BenchmarkManifestItem]:
    if target_clip_count <= 0:
        raise ValueError("target_clip_count must be positive")
    if prefix_seconds <= 0 or continuation_seconds <= 0:
        raise ValueError("prefix_seconds and continuation_seconds must be positive")

    required_duration = prefix_seconds + continuation_seconds
    excluded_fragments = tuple(fragment.lower() for fragment in excluded_tag_fragments)
    manifest: list[BenchmarkManifestItem] = []
    for record in records:
        if record.duration_seconds < required_duration:
            continue
        if not record.instrument_tags:
            continue
        if any(fragment in tag.lower() for tag in record.tags for fragment in excluded_fragments):
            continue
        license_value = licenses.get(record.track_id)
        if not license_value:
            continue
        manifest.append(
            BenchmarkManifestItem(
                track_id=record.track_id,
                source_dataset=DATASET_NAME,
                license=license_value,
                source_url=f"https://www.jamendo.com/track/{record.track_id}",
                audio_path=record.audio_path,
                duration_seconds=record.duration_seconds,
                prefix_window={"start_seconds": 0.0, "end_seconds": float(prefix_seconds)},
                generation_window={
                    "start_seconds": float(prefix_seconds),
                    "end_seconds": float(required_duration),
                },
                tags=record.tags,
                instrument_tags=record.instrument_tags,
            )
        )
        if len(manifest) >= target_clip_count:
            break

    if len(manifest) < target_clip_count:
        raise ValueError(
            "not enough eligible real MTG-Jamendo tracks for requested target_clip_count "
            f"({len(manifest)} found, {target_clip_count} requested)"
        )
    return manifest


def write_manifest_jsonl(items: Iterable[BenchmarkManifestItem], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            json.dump(item.to_dict(), f, sort_keys=True)
            f.write("\n")
    return output_path


def require_existing_file(path: str | Path, description: str) -> Path:
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved
