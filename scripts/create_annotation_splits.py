#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicgen_exp.annotation_workflow import create_verified_splits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic splits from verified annotations.")
    parser.add_argument("--annotations", required=True, help="Verified annotation JSONL.")
    parser.add_argument("--schema", default="schemas/annotation.schema.json")
    parser.add_argument("--output", default="data/annotation_splits.json")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    args = parser.parse_args()

    try:
        output_path = create_verified_splits(
            annotations_path=args.annotations,
            schema_path=args.schema,
            output_path=args.output,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote annotation splits to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
