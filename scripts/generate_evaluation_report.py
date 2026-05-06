#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicgen_exp.evaluation import generate_evaluation_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate evaluation summaries and dashboards.")
    parser.add_argument("--probe-results", required=True)
    parser.add_argument("--intervention-index", required=True)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    try:
        dashboard_path = generate_evaluation_report(
            probe_results_path=args.probe_results,
            intervention_index_path=args.intervention_index,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote evaluation dashboard to {dashboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
