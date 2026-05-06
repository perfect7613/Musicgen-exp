from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from musicgen_exp.config import ExperimentConfig


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at_utc: str
    git_commit: str
    config_path: str
    project_name: str
    seed: int
    outputs: dict[str, str]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_run_manifest(
    config: ExperimentConfig,
    output_dir: str | Path,
    run_id: str | None = None,
) -> Path:
    created_at = datetime.now(timezone.utc)
    resolved_run_id = run_id or f"{config.project_name}-{created_at.strftime('%Y%m%dT%H%M%SZ')}"
    manifest = RunManifest(
        run_id=resolved_run_id,
        created_at_utc=created_at.isoformat(),
        git_commit=current_git_commit(config.path.parent),
        config_path=str(config.path),
        project_name=config.project_name,
        seed=config.seed,
        outputs=config.output_roots,
        config=config.raw,
    )

    manifest_dir = Path(output_dir) / resolved_run_id
    manifest_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = manifest_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


def current_git_commit(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()
