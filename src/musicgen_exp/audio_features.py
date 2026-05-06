from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureArtifact:
    track_id: str
    feature_path: str
    sample_rate: int
    hop_length: int
    duration_seconds: float
    rms_mean: float
    spectral_centroid_mean: float
    tempo_bpm: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecurrenceProposal:
    track_id: str
    motif_window: dict[str, float]
    recurrence_window: dict[str, float]
    dtw_distance: float
    feature_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_chroma_artifact(
    audio_path: str | Path,
    output_dir: str | Path,
    track_id: str,
    sample_rate: int = 32000,
    hop_length: int = 512,
) -> FeatureArtifact:
    resolved_audio_path = require_existing_audio(audio_path)
    librosa = import_librosa()

    audio, loaded_sample_rate = librosa.load(resolved_audio_path, sr=sample_rate, mono=True)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=loaded_sample_rate, hop_length=hop_length)
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=loaded_sample_rate, hop_length=hop_length)
    tempo = estimate_tempo(audio, loaded_sample_rate, hop_length)

    output_path = Path(output_dir) / f"{track_id}.features.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        chroma=chroma,
        rms=rms,
        spectral_centroid=centroid,
        sample_rate=loaded_sample_rate,
        hop_length=hop_length,
    )

    return FeatureArtifact(
        track_id=track_id,
        feature_path=str(output_path),
        sample_rate=int(loaded_sample_rate),
        hop_length=int(hop_length),
        duration_seconds=float(len(audio) / loaded_sample_rate),
        rms_mean=float(np.mean(rms)),
        spectral_centroid_mean=float(np.mean(centroid)),
        tempo_bpm=tempo,
    )


def propose_recurrence_from_features(
    track_id: str,
    feature_path: str | Path,
    motif_window: dict[str, float],
    search_window: dict[str, float],
    proposal_seconds: float,
) -> RecurrenceProposal:
    librosa = import_librosa()
    features = np.load(feature_path)
    chroma = features["chroma"]
    sample_rate = int(features["sample_rate"])
    hop_length = int(features["hop_length"])

    motif_slice = time_window_to_frame_slice(motif_window, sample_rate, hop_length, chroma.shape[1])
    motif = chroma[:, motif_slice]
    if motif.shape[1] < 2:
        raise ValueError(f"{track_id}: motif window is too short for chroma/DTW matching")

    search_start = float(search_window["start_seconds"])
    search_end = float(search_window["end_seconds"])
    if proposal_seconds <= 0 or search_end <= search_start:
        raise ValueError(f"{track_id}: invalid recurrence search window")

    best_distance: float | None = None
    best_start = search_start
    step_seconds = frames_to_seconds(1, sample_rate, hop_length)
    current = search_start
    while current + proposal_seconds <= search_end:
        candidate_window = {
            "start_seconds": current,
            "end_seconds": current + proposal_seconds,
        }
        candidate_slice = time_window_to_frame_slice(
            candidate_window, sample_rate, hop_length, chroma.shape[1]
        )
        candidate = chroma[:, candidate_slice]
        if candidate.shape[1] >= 2:
            distance = float(librosa.sequence.dtw(X=motif, Y=candidate, metric="cosine")[0][-1, -1])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_start = current
        current += step_seconds

    if best_distance is None:
        raise ValueError(f"{track_id}: no valid recurrence candidates found")

    return RecurrenceProposal(
        track_id=track_id,
        motif_window=motif_window,
        recurrence_window={
            "start_seconds": float(best_start),
            "end_seconds": float(best_start + proposal_seconds),
        },
        dtw_distance=best_distance,
        feature_path=str(feature_path),
    )


def load_manifest_jsonl(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = require_existing_file(path, "benchmark manifest JSONL")
    items: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}:{line_number}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{manifest_path}:{line_number}: expected JSON object")
            items.append(item)
    return items


def write_jsonl(items: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            json.dump(item, f, sort_keys=True)
            f.write("\n")
    return output_path


def resolve_manifest_audio_path(audio_root: str | Path, manifest_item: dict[str, Any]) -> Path:
    audio_path = manifest_item.get("audio_path")
    if not isinstance(audio_path, str) or not audio_path:
        raise ValueError("manifest item is missing audio_path")
    return Path(audio_root) / audio_path


def require_existing_audio(path: str | Path) -> Path:
    return require_existing_file(path, "real benchmark audio")


def require_existing_file(path: str | Path, description: str) -> Path:
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def import_librosa() -> Any:
    try:
        import librosa
    except ModuleNotFoundError as exc:
        raise RuntimeError("install audio dependencies with `uv sync --extra audio --extra dev`") from exc
    return librosa


def estimate_tempo(audio: np.ndarray, sample_rate: int, hop_length: int) -> float | None:
    librosa = import_librosa()
    try:
        tempo = librosa.beat.tempo(y=audio, sr=sample_rate, hop_length=hop_length)
    except Exception:
        return None
    if len(tempo) == 0:
        return None
    return float(tempo[0])


def time_window_to_frame_slice(
    window: dict[str, float],
    sample_rate: int,
    hop_length: int,
    max_frames: int,
) -> slice:
    start = seconds_to_frame(float(window["start_seconds"]), sample_rate, hop_length)
    end = seconds_to_frame(float(window["end_seconds"]), sample_rate, hop_length)
    start = max(0, min(start, max_frames))
    end = max(start, min(end, max_frames))
    return slice(start, end)


def seconds_to_frame(seconds: float, sample_rate: int, hop_length: int) -> int:
    return int(round(seconds * sample_rate / hop_length))


def frames_to_seconds(frames: int, sample_rate: int, hop_length: int) -> float:
    return float(frames * hop_length / sample_rate)
