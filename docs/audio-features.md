# Real-Audio Motif Features

This stage extracts chroma features and motif-recurrence proposals from real benchmark audio only.

## Inputs

- A real benchmark manifest produced by `scripts/build_benchmark_manifest.py`.
- A local MTG-Jamendo audio root containing the referenced audio paths.

The command fails when either input is missing.

## Install Audio Dependencies

```bash
uv sync --extra dev --extra audio
```

## Run

```bash
uv run python scripts/extract_motif_features.py \
  --manifest data/benchmark_manifest.jsonl \
  --audio-root /path/to/mtg-jamendo-audio \
  --features-dir outputs/features \
  --proposals data/recurrence_proposals.jsonl
```

## Outputs

- Compressed feature artifacts containing chroma, RMS, spectral centroid, sample rate, and hop length.
- Recurrence proposal JSONL rows with motif window, proposed recurrence window, and DTW distance.

These are proposals only. They must be manually verified before final experiments.
