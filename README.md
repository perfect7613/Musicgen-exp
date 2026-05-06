# MusicGen Long-Horizon Coherence Experiment

Classic mechanistic interpretability study testing whether MusicGen contains SAE features that predict and causally affect long-horizon motif recurrence.

Public framing:

> Do autoregressive music models have foresight-like coherence features?

This repo starts from the MusicGen + SAE workflow in PapayaResearch `musicdiscovery` and extends it with future-event probes, held-out causal interventions, and motif-recurrence evaluation on Creative Commons instrumental music.

This project follows a strict [real-data-only policy](docs/data-policy.md): no fake musical benchmark records, synthetic motif fixtures, placeholder annotations, or fake model outputs should be committed as experiment evidence.

## Research Question

Do any MusicGen residual-stream / SAE features causally influence future musical structure, especially motif recurrence, beyond local fluency or position/time shortcuts?

## V1 Scope

- Model: MusicGen-Small for debugging, MusicGen-Large for main results when compute allows.
- SAE source: existing `musicdiscovery` checkpoints first.
- Dataset source: MTG-Jamendo instrumental Creative Commons tracks.
- Clip format: 10s real-music prefix + 30s MusicGen continuation.
- Primary event: motif recurrence.
- Primary metric: chroma similarity + Dynamic Time Warping.
- Main intervention: SAE feature ablation/scaling during generation.

## Repository Layout

```text
configs/              Experiment configs
data/annotations/     Benchmark annotation examples and generated labels
docs/                 Project plan and RunPod workflow
notebooks/            Exploratory notebooks
outputs/              Figures, audio demos, and run artifacts
schemas/              JSON schemas for benchmark annotations
scripts/              Setup, validation, and RunPod helper scripts
src/musicgen_exp/     Reusable experiment code
```

## Quick Start

```bash
uv sync --extra dev
uv run python scripts/validate_annotations.py --schema-only
uv run python scripts/validate_config.py --config configs/experiment.yaml
uv run pytest
```

## RunPod

RunPod setup is documented in [docs/runpod.md](docs/runpod.md). The repo includes helper scripts, but pod creation is intentionally manual/explicit so credits are not spent accidentally.

## Evidence Standard

A candidate feature only counts as a long-horizon coherence feature if it:

- Predicts held-out future motif recurrence above position/acoustic controls.
- Shows stronger long-horizon signal than local-only signal.
- Causally changes future motif recurrence when ablated/scaled.
- Does not merely degrade local audio quality, loudness, tempo, or texture.

Null results are publishable if the controls and failure modes are documented clearly.
