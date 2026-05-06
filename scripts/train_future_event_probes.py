#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicgen_exp.config import load_config  # noqa: E402
from musicgen_exp.probes import train_probe_suite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train future-event probes on real activation artifacts.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--activation-index", required=True)
    parser.add_argument("--annotation-splits", required=True)
    parser.add_argument("--output-dir", default="outputs/probes")
    parser.add_argument("--activation-kind", choices=["residual", "sae"], default="residual")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        horizon_names = list(config.raw["horizons"].keys())
        results_path = train_probe_suite(
            activation_index_path=args.activation_index,
            annotation_splits_path=args.annotation_splits,
            output_dir=args.output_dir,
            horizon_names=horizon_names,
            activation_kind=args.activation_kind,
        )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote probe results to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
