# Pipeline

The experiment is organized as restartable stages. Every stage should be driven by an explicit config and should write auditable artifacts.

## Stage 0: Validate Configuration

```bash
uv run python scripts/validate_config.py --config configs/experiment.yaml
```

This checks the structural config required before any data, model, or GPU work begins.

## Stage 1: Create A Run Manifest

```bash
uv run python scripts/create_run_manifest.py \
  --config configs/experiment.yaml \
  --output-dir outputs/runs
```

The manifest records:

- The config used for the run.
- The git commit.
- The UTC creation time.
- The output roots declared by the config.

## Data Integrity

Pipeline stages must not create fake benchmark items, synthetic music examples, or fake model outputs. Stages that need real MTG-Jamendo metadata, real audio, or real MusicGen activations must fail clearly when those inputs are absent.

## Stage 2: Build Benchmark Manifest

After downloading official MTG-Jamendo metadata and `audio_licenses.txt`, build the manifest:

```bash
uv run python scripts/build_benchmark_manifest.py \
  --metadata-tsv external/mtg-jamendo-dataset/data/autotagging_instrument.tsv \
  --audio-licenses external/mtg-jamendo-dataset/audio_licenses.txt \
  --output data/benchmark_manifest.jsonl
```

The command fails if the real metadata/license files are missing.

## Stage 3: Extract Motif Features

After the manifest exists and the referenced MTG-Jamendo audio files are available locally:

```bash
uv sync --extra dev --extra audio
uv run python scripts/extract_motif_features.py \
  --manifest data/benchmark_manifest.jsonl \
  --audio-root /path/to/mtg-jamendo-audio \
  --features-dir outputs/features \
  --proposals data/recurrence_proposals.jsonl
```

The command fails if the manifest or real audio files are missing.

## Stage 4: Manual Review And Splits

Prepare a review queue from real recurrence proposals:

```bash
uv run python scripts/prepare_annotation_review.py \
  --manifest data/benchmark_manifest.jsonl \
  --proposals data/recurrence_proposals.jsonl \
  --output data/annotations/review_queue.jsonl
```

After human review, create deterministic splits:

```bash
uv run python scripts/create_annotation_splits.py \
  --annotations data/annotations/verified.jsonl \
  --schema schemas/annotation.schema.json \
  --output data/annotation_splits.json
```

## Stage 5: Extract MusicGen Activations

On RunPod, install musicdiscovery and model dependencies, then extract residual and optional SAE activations:

```bash
uv run python scripts/extract_musicgen_activations.py \
  --mode teacher-forced \
  --manifest data/benchmark_manifest.jsonl \
  --audio-root /path/to/mtg-jamendo-audio \
  --musicdiscovery-path /workspace/musicdiscovery \
  --model-size pilot \
  --device cuda \
  --output-dir outputs/activations
```

See `docs/musicgen-integration.md`.

## Stage 6: Train Future-Event Probes

After teacher-forced activations and verified annotation splits exist:

```bash
uv sync --extra dev --extra analysis
uv run python scripts/train_future_event_probes.py \
  --activation-index outputs/activations/teacher_forced/activation_index.jsonl \
  --annotation-splits data/annotation_splits.json \
  --activation-kind residual \
  --output-dir outputs/probes/residual
```

See `docs/probes.md`.

## Stage 7: Run SAE Interventions

After candidate features are ranked and SAE checkpoints are available:

```bash
uv run python scripts/run_sae_interventions.py \
  --manifest data/benchmark_manifest.jsonl \
  --candidate-ranking outputs/probes/sae/local_seconds.candidate_feature_ranking.jsonl \
  --audio-root /path/to/mtg-jamendo-audio \
  --musicdiscovery-path /workspace/musicdiscovery \
  --sae-checkpoint-root /path/to/sae/checkpoints \
  --model-size primary \
  --device cuda \
  --seeds 7613,7614,7615 \
  --top-k 5 \
  --output-dir outputs/interventions
```

See `docs/interventions.md`.
