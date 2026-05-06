#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicgen_exp.config import load_config  # noqa: E402
from musicgen_exp.run_manifest import create_run_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a run manifest for a configured experiment.")
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to the experiment config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/runs",
        help="Directory where the run manifest directory should be created.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = create_run_manifest(config, args.output_dir)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
