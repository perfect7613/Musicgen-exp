#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicgen_exp.annotation_workflow import build_review_queue  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare manual review rows from real proposals.")
    parser.add_argument("--manifest", required=True, help="Benchmark manifest JSONL.")
    parser.add_argument("--proposals", required=True, help="Recurrence proposal JSONL.")
    parser.add_argument(
        "--output",
        default="data/annotations/review_queue.jsonl",
        help="Manual review queue output path.",
    )
    args = parser.parse_args()

    try:
        output_path = build_review_queue(args.manifest, args.proposals, args.output)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote manual review queue to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
