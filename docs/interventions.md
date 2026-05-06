# SAE Feature Interventions

Interventions require real candidate feature rankings, real benchmark audio, real MusicGen generation, and real SAE checkpoints.

Candidate ranking rows must include:

- `feature_index`
- `layer`, or pass `--layer` when running a single-layer intervention sweep
- Optional `random_control_feature_index`
- Optional `local_control_feature_index`

## Run

```bash
uv run python scripts/run_sae_interventions.py \
  --manifest data/benchmark_manifest.jsonl \
  --candidate-ranking outputs/probes/sae/local_seconds.candidate_feature_ranking.jsonl \
  --audio-root /path/to/mtg-jamendo-audio \
  --musicdiscovery-path /workspace/musicdiscovery \
  --sae-checkpoint-root /path/to/sae/checkpoints \
  --model-size primary \
  --device cuda \
  --layer 24 \
  --seeds 7613,7614,7615 \
  --top-k 5 \
  --output-dir outputs/interventions
```

## Conditions

- `baseline`
- `ablate`
- `scale`
- `random_feature`
- `local_feature`

Generated audio is stored as local `.npy` tensors so source audio is not rehosted. The evaluation slice computes recurrence-change metrics from these outputs.
