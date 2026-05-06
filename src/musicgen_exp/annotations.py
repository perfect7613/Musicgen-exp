from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ModuleNotFoundError:  # pragma: no cover - exercised only before dependencies are installed.
    Draft202012Validator = None


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected object at {path}, got {type(data).__name__}")
    return data


def validate_annotation(annotation_path: str | Path, schema_path: str | Path) -> list[str]:
    annotation = load_json(annotation_path)
    schema = load_json(schema_path)
    if Draft202012Validator is None:
        return validate_annotation_minimal(annotation, schema)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(annotation), key=lambda err: list(err.path))
    return [format_error(error) for error in errors]


def format_error(error: Any) -> str:
    location = ".".join(str(part) for part in error.path) or "<root>"
    return f"{location}: {error.message}"


def validate_annotation_minimal(annotation: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Fallback checks for fresh environments before jsonschema is installed."""
    errors: list[str] = []
    for field in schema.get("required", []):
        if field not in annotation:
            errors.append(f"{field}: required property is missing")

    for field in ("prefix_window", "generation_window", "motif_window", "recurrence_window"):
        value = annotation.get(field)
        if not isinstance(value, dict):
            errors.append(f"{field}: expected object")
            continue
        start = value.get("start_seconds")
        end = value.get("end_seconds")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            errors.append(f"{field}: start_seconds and end_seconds must be numbers")
        elif end <= start:
            errors.append(f"{field}: end_seconds must be greater than start_seconds")

    return errors
