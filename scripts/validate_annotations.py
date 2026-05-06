#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicgen_exp.annotations import validate_annotation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark annotation JSON files.")
    parser.add_argument("paths", nargs="+", help="Annotation JSON files to validate.")
    parser.add_argument(
        "--schema",
        default="schemas/annotation.schema.json",
        help="Path to annotation JSON schema.",
    )
    args = parser.parse_args()

    schema_path = Path(args.schema)
    had_errors = False
    for raw_path in args.paths:
        annotation_path = Path(raw_path)
        errors = validate_annotation(annotation_path, schema_path)
        if errors:
            had_errors = True
            print(f"{annotation_path}: invalid", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
        else:
            print(f"{annotation_path}: ok")

    return 1 if had_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
