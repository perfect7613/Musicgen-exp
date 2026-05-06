# Future-Event Probes

Probe training requires real activation artifacts and manually verified annotation splits.

## Install Analysis Dependencies

```bash
uv sync --extra dev --extra analysis
```

## Train

```bash
uv run python scripts/train_future_event_probes.py \
  --activation-index outputs/activations/teacher_forced/activation_index.jsonl \
  --annotation-splits data/annotation_splits.json \
  --activation-kind residual \
  --output-dir outputs/probes/residual
```

For SAE feature probes:

```bash
uv run python scripts/train_future_event_probes.py \
  --activation-index outputs/activations/teacher_forced/activation_index.jsonl \
  --annotation-splits data/annotation_splits.json \
  --activation-kind sae \
  --output-dir outputs/probes/sae
```

## Controls

The runner emits results for:

- Main activation probe.
- Position-only control.
- Acoustic-only control.
- Prompt/source control.
- Shuffled-label control.

Candidate feature rankings are produced only from the training split for the main activation probe.
