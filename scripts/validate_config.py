#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicgen_exp.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an experiment config.")
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to the experiment config.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"{config.path}: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
