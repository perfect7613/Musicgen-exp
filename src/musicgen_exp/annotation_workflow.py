from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from musicgen_exp.audio_features import load_manifest_jsonl, require_existing_file, write_jsonl
from musicgen_exp.annotations import load_json, validate_annotation_object


VERIFIED_STATUSES = {"verified_positive", "verified_negative", "ambiguous", "rejected"}
SPLIT_NAMES = ("train", "validation", "test")


def build_review_queue(
    manifest_path: str | Path,
    proposals_path: str | Path,
    output_path: str | Path,
) -> Path:
    manifest_items = {str(item["track_id"]): item for item in load_manifest_jsonl(manifest_path)}
    proposal_items = load_manifest_jsonl(proposals_path)
    review_rows: list[dict[str, Any]] = []

    for proposal in proposal_items:
        track_id = str(proposal["track_id"])
        manifest_item = manifest_items.get(track_id)
        if manifest_item is None:
            raise ValueError(f"{track_id}: proposal has no matching benchmark manifest item")
        review_rows.append(
            {
                "track_id": track_id,
                "source_dataset": manifest_item["source_dataset"],
                "license": manifest_item["license"],
                "source_url": manifest_item["source_url"],
                "prompt": manifest_item.get("prompt", ""),
                "prefix_window": manifest_item["prefix_window"],
                "generation_window": manifest_item["generation_window"],
                "motif_window": proposal["motif_window"],
                "recurrence_window": proposal["recurrence_window"],
                "event_type": "ambiguous",
                "instrument_tags": manifest_item.get("instrument_tags", []),
                "section_label": "unreviewed",
                "energy": proposal.get("feature_summary", {}).get("rms_mean", 0.0),
                "spectral_centroid": proposal.get("feature_summary", {}).get(
                    "spectral_centroid_mean", 0.0
                ),
                "chroma_features_path": proposal.get("feature_path", ""),
                "manual_verification_status": "unverified",
                "notes": "Requires human review before final experiment use.",
            }
        )

    return write_jsonl(review_rows, output_path)


def create_verified_splits(
    annotations_path: str | Path,
    schema_path: str | Path,
    output_path: str | Path,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> Path:
    annotations = load_annotation_jsonl(annotations_path)
    schema = load_json(schema_path)
    split_rows = {split_name: [] for split_name in SPLIT_NAMES}

    for annotation in annotations:
        validation_errors = validate_annotation_object(annotation, schema)
        if validation_errors:
            raise ValueError(
                f"{annotation.get('track_id', '<unknown>')}: invalid annotation: "
                + "; ".join(validation_errors)
            )
        status = annotation["manual_verification_status"]
        if status not in VERIFIED_STATUSES:
            raise ValueError(f"{annotation['track_id']}: annotation is not manually verified")
        if status == "rejected":
            continue
        split_name = assign_split(str(annotation["track_id"]), train_ratio, validation_ratio)
        split_rows[split_name].append(annotation)

    output = {
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "test_ratio": 1.0 - train_ratio - validation_ratio,
        "splits": split_rows,
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    return output_file


def load_annotation_jsonl(path: str | Path) -> list[dict[str, Any]]:
    annotation_path = require_existing_file(path, "annotation JSONL")
    rows: list[dict[str, Any]] = []
    with annotation_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{annotation_path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{annotation_path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def assign_split(track_id: str, train_ratio: float, validation_ratio: float) -> str:
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("split ratios must be positive and sum to less than 1")
    digest = hashlib.sha256(track_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + validation_ratio:
        return "validation"
    return "test"
