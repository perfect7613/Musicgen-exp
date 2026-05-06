from __future__ import annotations

import json
from pathlib import Path

from musicgen_exp.annotations import load_json, validate_annotation_object


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "annotation.schema.json"


def test_annotation_schema_declares_required_provenance_fields() -> None:
    schema = load_json(SCHEMA_PATH)

    required = set(schema["required"])

    assert "track_id" in required
    assert "source_dataset" in required
    assert "license" in required
    assert "source_url" in required
    assert "manual_verification_status" in required


def test_empty_annotation_is_rejected() -> None:
    schema = load_json(SCHEMA_PATH)

    errors = validate_annotation_object({}, schema)

    assert errors
    assert any("track_id" in error for error in errors)


def test_schema_file_is_json() -> None:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        parsed = json.load(f)

    assert parsed["title"] == "MusicGen Long-Horizon Coherence Annotation"
