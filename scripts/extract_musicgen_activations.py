#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicgen_exp.config import load_config  # noqa: E402
from musicgen_exp.model_integration import (  # noqa: E402
    ModelSpec,
    build_sae_specs,
    extract_real_audio_activations,
    extract_real_generation_activations,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract real MusicGen residual/SAE activations from benchmark audio."
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--manifest", required=True, help="Benchmark manifest JSONL.")
    parser.add_argument("--audio-root", required=True, help="Root directory containing real audio.")
    parser.add_argument("--output-dir", default="outputs/activations")
    parser.add_argument("--musicdiscovery-path", default=None)
    parser.add_argument("--sae-checkpoint-root", default=None)
    parser.add_argument("--model-size", choices=["pilot", "primary"], default="pilot")
    parser.add_argument(
        "--mode",
        choices=["teacher-forced", "generation"],
        default="teacher-forced",
        help="Activation extraction mode.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        model_name = str(config.raw["model"][args.model_size])
        layers = list(config.raw["model"]["primary_layers"])
        if args.model_size == "pilot":
            layers = list(config.raw["model"].get("pilot_layers", layers))
        model_spec = ModelSpec(
            model_name=model_name,
            device=args.device,
            precision=args.precision,
            layers=[int(layer) for layer in layers],
            musicdiscovery_path=args.musicdiscovery_path,
        )
        if args.mode == "teacher-forced":
            sae_specs = build_sae_specs(args.sae_checkpoint_root, model_spec.layers)
            index_path = extract_real_audio_activations(
                manifest_path=args.manifest,
                audio_root=args.audio_root,
                output_dir=args.output_dir,
                model_spec=model_spec,
                sae_specs=sae_specs,
                limit=args.limit,
            )
        else:
            if args.sae_checkpoint_root is not None:
                raise ValueError("generation mode currently captures residual hooks; omit SAE checkpoint root")
            index_path = extract_real_generation_activations(
                manifest_path=args.manifest,
                audio_root=args.audio_root,
                output_dir=args.output_dir,
                model_spec=model_spec,
                limit=args.limit,
            )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote activation index to {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
