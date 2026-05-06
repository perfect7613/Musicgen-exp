from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def project_name(self) -> str:
        return str(self.raw["project"]["name"])

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    @property
    def output_roots(self) -> dict[str, str]:
        return dict(self.raw["outputs"])


REQUIRED_PATHS: tuple[tuple[str, ...], ...] = (
    ("project", "name"),
    ("project", "seed"),
    ("model", "pilot"),
    ("model", "primary"),
    ("model", "primary_layers"),
    ("sae", "source"),
    ("benchmark", "source"),
    ("benchmark", "target_clip_count"),
    ("benchmark", "prefix_seconds"),
    ("benchmark", "continuation_seconds"),
    ("horizons", "local_seconds"),
    ("horizons", "medium_seconds"),
    ("horizons", "long_seconds"),
    ("metrics", "primary_motif_metric"),
    ("metrics", "controls"),
    ("outputs", "figures_dir"),
    ("outputs", "audio_dir"),
)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path}: expected a YAML object")

    errors = validate_config_object(loaded)
    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"{config_path}: invalid experiment config\n{formatted}")

    return ExperimentConfig(raw=loaded, path=config_path)


def validate_config_object(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for required_path in REQUIRED_PATHS:
        if lookup_path(config, required_path) is None:
            errors.append(f"{'.'.join(required_path)} is required")

    errors.extend(validate_positive_int(config, ("project", "seed")))
    errors.extend(validate_positive_int(config, ("benchmark", "target_clip_count")))
    errors.extend(validate_positive_number(config, ("benchmark", "prefix_seconds")))
    errors.extend(validate_positive_number(config, ("benchmark", "continuation_seconds")))
    errors.extend(validate_number_pair(config, ("horizons", "local_seconds")))
    errors.extend(validate_number_pair(config, ("horizons", "medium_seconds")))
    errors.extend(validate_number_pair(config, ("horizons", "long_seconds")))
    return errors


def lookup_path(config: dict[str, Any], path: tuple[str, ...]) -> Any | None:
    current: Any = config
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def validate_positive_int(config: dict[str, Any], path: tuple[str, ...]) -> list[str]:
    value = lookup_path(config, path)
    if value is None:
        return []
    if not isinstance(value, int) or value < 0:
        return [f"{'.'.join(path)} must be a non-negative integer"]
    return []


def validate_positive_number(config: dict[str, Any], path: tuple[str, ...]) -> list[str]:
    value = lookup_path(config, path)
    if value is None:
        return []
    if not isinstance(value, (int, float)) or value <= 0:
        return [f"{'.'.join(path)} must be a positive number"]
    return []


def validate_number_pair(config: dict[str, Any], path: tuple[str, ...]) -> list[str]:
    value = lookup_path(config, path)
    if value is None:
        return []
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, (int, float)) for item in value)
    ):
        return [f"{'.'.join(path)} must be a two-number list"]
    if value[0] >= value[1]:
        return [f"{'.'.join(path)} must be ordered from low to high"]
    return []
