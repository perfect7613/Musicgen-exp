from __future__ import annotations

from pathlib import Path

import pytest

from musicgen_exp.config import load_config, validate_config_object
from musicgen_exp.run_manifest import create_run_manifest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"


def test_repository_config_loads() -> None:
    config = load_config(CONFIG_PATH)

    assert config.project_name == "musicgen-long-horizon-coherence"
    assert config.seed == 7613
    assert config.raw["benchmark"]["source"] == "mtg-jamendo"


def test_empty_config_is_rejected() -> None:
    errors = validate_config_object({})

    assert errors
    assert any("project.name" in error for error in errors)
    assert any("benchmark.source" in error for error in errors)


def test_invalid_horizon_order_is_rejected() -> None:
    config = load_config(CONFIG_PATH).raw
    config["horizons"]["long_seconds"] = [25, 15]

    errors = validate_config_object(config)

    assert "horizons.long_seconds must be ordered from low to high" in errors


def test_run_manifest_records_config_and_git_commit(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)

    manifest_path = create_run_manifest(config, tmp_path)

    assert manifest_path.name == "run_manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "musicgen-long-horizon-coherence" in manifest_text
    assert "git_commit" in manifest_text


def test_run_manifest_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    create_run_manifest(config, tmp_path, run_id="fixed-run-id")

    with pytest.raises(FileExistsError):
        create_run_manifest(config, tmp_path, run_id="fixed-run-id")
