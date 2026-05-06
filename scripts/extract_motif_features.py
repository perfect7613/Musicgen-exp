#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicgen_exp.audio_features import (  # noqa: E402
    extract_chroma_artifact,
    load_manifest_jsonl,
    propose_recurrence_from_features,
    resolve_manifest_audio_path,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract real-audio chroma features and recurrence proposals."
    )
    parser.add_argument("--manifest", required=True, help="Benchmark manifest JSONL.")
    parser.add_argument("--audio-root", required=True, help="Root directory containing real audio files.")
    parser.add_argument("--features-dir", default="outputs/features")
    parser.add_argument("--proposals", default="data/recurrence_proposals.jsonl")
    parser.add_argument("--failures", default="data/recurrence_failures.jsonl")
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        manifest = load_manifest_jsonl(args.manifest)
        selected_items = manifest[: args.limit] if args.limit is not None else manifest
        proposals: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        for item in selected_items:
            track_id = str(item["track_id"])
            try:
                audio_path = resolve_manifest_audio_path(args.audio_root, item)
                artifact = extract_chroma_artifact(
                    audio_path=audio_path,
                    output_dir=args.features_dir,
                    track_id=track_id,
                    sample_rate=args.sample_rate,
                    hop_length=args.hop_length,
                )
                prefix_window = item["prefix_window"]
                generation_window = item["generation_window"]
                motif_window = {
                    "start_seconds": float(prefix_window["start_seconds"]),
                    "end_seconds": float(prefix_window["end_seconds"]),
                }
                proposal_seconds = motif_window["end_seconds"] - motif_window["start_seconds"]
                proposal = propose_recurrence_from_features(
                    track_id=track_id,
                    feature_path=artifact.feature_path,
                    motif_window=motif_window,
                    search_window=generation_window,
                    proposal_seconds=proposal_seconds,
                )
                proposal_row = proposal.to_dict()
                proposal_row["feature_summary"] = artifact.to_dict()
                proposals.append(proposal_row)
            except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
                failures.append({"track_id": track_id, "error": str(exc)})
        output_path = write_jsonl(proposals, args.proposals)
        failures_path = write_jsonl(failures, args.failures)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote {len(proposals)} recurrence proposals to {output_path}")
    print(f"wrote {len(failures)} recurrence failures to {failures_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
