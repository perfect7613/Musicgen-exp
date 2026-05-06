#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicgen_exp.config import load_config  # noqa: E402
from musicgen_exp.interventions import run_intervention_suite  # noqa: E402
from musicgen_exp.model_integration import ModelSpec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real SAE feature interventions in MusicGen.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-ranking", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--musicdiscovery-path", required=True)
    parser.add_argument("--sae-checkpoint-root", required=True)
    parser.add_argument("--output-dir", default="outputs/interventions")
    parser.add_argument("--model-size", choices=["pilot", "primary"], default="pilot")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seeds", default="7613")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--layer", type=int, default=None, help="Layer override for non-layer-aware rankings.")
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
            precision="float32",
            layers=[int(layer) for layer in layers],
            musicdiscovery_path=args.musicdiscovery_path,
        )
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
        index_path = run_intervention_suite(
            manifest_path=args.manifest,
            candidate_ranking_path=args.candidate_ranking,
            audio_root=args.audio_root,
            output_dir=args.output_dir,
            model_spec=model_spec,
            sae_checkpoint_root=args.sae_checkpoint_root,
            strength=args.strength,
            seeds=seeds,
            top_k=args.top_k,
            layer_override=args.layer,
            limit=args.limit,
        )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote intervention index to {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
