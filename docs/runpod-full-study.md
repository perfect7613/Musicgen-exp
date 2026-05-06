# RunPod Full-Study Workflow

This workflow is budget-aware. Current observed balance on 2026-05-06 was `$24.51` with `$0/hr` current spend. Treat `$23` as the working cap unless the project owner explicitly raises it.

## Budget Gate

Before creating any pod:

```bash
runpodctl user
runpodctl pod list --all
runpodctl gpu list
```

Proceed only if:

- `currentSpendPerHr` is `0` before starting.
- No forgotten running pods exist.
- The selected GPU is available.
- The planned run duration fits the remaining balance.

Prefer short debugging runs on RTX 4090 or A40 before any A100/L40S run.

## Create A Debug Pod

Pod creation is intentionally manual. Do not automate this in tests.

```bash
runpodctl pod create \
  --name musicgen-exp-debug \
  --template-id runpod-torch-v220 \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --container-disk-in-gb 80 \
  --volume-in-gb 100 \
  --ports "8888/http,22/tcp"
```

Use A40 if 4090 availability is poor:

```bash
runpodctl pod create \
  --name musicgen-exp-a40 \
  --template-id runpod-torch-v220 \
  --gpu-id "NVIDIA A40" \
  --container-disk-in-gb 100 \
  --volume-in-gb 150 \
  --ports "8888/http,22/tcp"
```

## Transfer Repo

From local repo root:

```bash
bash scripts/package_for_runpod.sh
runpodctl send musicgen-exp.tar.gz
```

On the pod:

```bash
cd /workspace
# receive with the code printed by runpodctl send
tar -xzf musicgen-exp.tar.gz -C /workspace/musicgen-exp
```

## Pod Setup

```bash
cd /workspace/musicgen-exp
bash scripts/setup_remote.sh

cd /workspace
git clone https://github.com/PapayaResearch/musicdiscovery.git
cd musicdiscovery
pip install -r requirements.txt
```

## Data Placement

Keep restricted audio local to the pod volume. Do not commit or upload source audio.

Expected layout:

```text
/workspace/musicgen-exp/
/workspace/musicdiscovery/
/workspace/mtg-jamendo-dataset/
/workspace/mtg-jamendo-audio/
```

## Stage Order

Local or CPU-safe:

```bash
uv run python scripts/validate_config.py
uv run python scripts/validate_annotations.py --schema-only
uv run pytest
```

Metadata and audio stages:

```bash
uv run python scripts/build_benchmark_manifest.py \
  --metadata-tsv /workspace/mtg-jamendo-dataset/data/autotagging_instrument.tsv \
  --audio-licenses /workspace/mtg-jamendo-dataset/audio_licenses.txt \
  --output data/benchmark_manifest.jsonl

uv sync --extra dev --extra audio
uv run python scripts/extract_motif_features.py \
  --manifest data/benchmark_manifest.jsonl \
  --audio-root /workspace/mtg-jamendo-audio \
  --features-dir outputs/features \
  --proposals data/recurrence_proposals.jsonl
```

Manual review required:

```bash
uv run python scripts/prepare_annotation_review.py \
  --manifest data/benchmark_manifest.jsonl \
  --proposals data/recurrence_proposals.jsonl \
  --output data/annotations/review_queue.jsonl
```

Edit review statuses manually, then:

```bash
uv run python scripts/create_annotation_splits.py \
  --annotations data/annotations/verified.jsonl \
  --schema schemas/annotation.schema.json \
  --output data/annotation_splits.json
```

GPU-required:

```bash
uv run python scripts/extract_musicgen_activations.py \
  --mode teacher-forced \
  --manifest data/benchmark_manifest.jsonl \
  --audio-root /workspace/mtg-jamendo-audio \
  --musicdiscovery-path /workspace/musicdiscovery \
  --model-size pilot \
  --device cuda \
  --output-dir outputs/activations/teacher_forced
```

Analysis:

```bash
uv sync --extra dev --extra analysis
uv run python scripts/train_future_event_probes.py \
  --activation-index outputs/activations/teacher_forced/activation_index.jsonl \
  --annotation-splits data/annotation_splits.json \
  --activation-kind residual \
  --output-dir outputs/probes/residual
```

Interventions:

```bash
uv run python scripts/run_sae_interventions.py \
  --manifest data/benchmark_manifest.jsonl \
  --candidate-ranking outputs/probes/sae/local_seconds.candidate_feature_ranking.jsonl \
  --audio-root /workspace/mtg-jamendo-audio \
  --musicdiscovery-path /workspace/musicdiscovery \
  --sae-checkpoint-root /workspace/sae-checkpoints \
  --model-size primary \
  --device cuda \
  --layer 24 \
  --seeds 7613,7614,7615 \
  --top-k 5 \
  --output-dir outputs/interventions
```

Evaluation:

```bash
uv run python scripts/generate_evaluation_report.py \
  --probe-results outputs/probes/sae/probe_results.jsonl \
  --intervention-index outputs/interventions/intervention_index.jsonl \
  --output-dir outputs/evaluation
```

## Artifact Transfer

Transfer only safe artifacts:

```bash
tar -czf musicgen-exp-artifacts.tar.gz \
  data/benchmark_manifest.jsonl \
  data/annotations \
  data/annotation_splits.json \
  outputs/features \
  outputs/activations \
  outputs/probes \
  outputs/interventions \
  outputs/evaluation

runpodctl send musicgen-exp-artifacts.tar.gz
```

Do not transfer or publish source MTG-Jamendo audio unless every license is explicitly verified.

## Shutdown

Immediately after transfer:

```bash
runpodctl pod list
```

Then stop/terminate the pod in the RunPod console or with the appropriate `runpodctl pod` command after confirming no long job is running.

Re-check:

```bash
runpodctl user
runpodctl pod list --all
```

The target after shutdown is `currentSpendPerHr: 0`.

## Release Checklist

- Public repo contains code, schemas, docs, configs, and non-restricted metadata only.
- Source audio is not committed.
- Generated audio is only published when allowed by the benchmark source licenses and model-output policy.
- `evaluation_summary.json` and `feature_dashboard.md` are generated from real artifacts.
- Alignment Forum / LessWrong draft states positive, mixed, or null result honestly.
- Claims use “long-horizon coherence features” unless causal evidence is unusually strong.
