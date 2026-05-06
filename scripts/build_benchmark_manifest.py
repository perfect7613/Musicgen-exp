#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicgen_exp.config import load_config  # noqa: E402
from musicgen_exp.mtg_jamendo import (  # noqa: E402
    build_benchmark_manifest,
    read_audio_licenses,
    read_autotagging_tsv,
    write_manifest_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a benchmark manifest from real MTG-Jamendo metadata."
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--metadata-tsv",
        required=True,
        help="Path to an official MTG-Jamendo metadata TSV, e.g. data/autotagging_instrument.tsv.",
    )
    parser.add_argument(
        "--audio-licenses",
        required=True,
        help="Path to official MTG-Jamendo audio_licenses.txt.",
    )
    parser.add_argument(
        "--output",
        default="data/benchmark_manifest.jsonl",
        help="Path where the benchmark manifest JSONL should be written.",
    )
    parser.add_argument(
        "--audio-root",
        default=None,
        help="Optional root of downloaded real audio; when set, only existing audio files are included.",
    )
    parser.add_argument(
        "--audio-variant",
        choices=["original", "audio-low"],
        default="original",
        help="Audio filename variant to write into the manifest.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        benchmark_config = config.raw["benchmark"]
        records = read_autotagging_tsv(args.metadata_tsv)
        licenses = read_audio_licenses(args.audio_licenses)
        manifest = build_benchmark_manifest(
            records=records,
            licenses=licenses,
            prefix_seconds=float(benchmark_config["prefix_seconds"]),
            continuation_seconds=float(benchmark_config["continuation_seconds"]),
            target_clip_count=int(benchmark_config["target_clip_count"]),
            audio_root=args.audio_root,
            audio_variant=args.audio_variant,
        )
        output_path = write_manifest_jsonl(manifest, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote {len(manifest)} real MTG-Jamendo manifest items to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
