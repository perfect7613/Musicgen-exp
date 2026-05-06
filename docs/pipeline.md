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
