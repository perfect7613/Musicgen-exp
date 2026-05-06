# MusicGen / musicdiscovery Integration

This project uses the upstream `musicdiscovery` `HookedMusicGen` implementation for classic mechanistic interpretability hooks.

`HookedMusicGen` already uses TransformerLens primitives:

- `HookedRootModule`
- `HookPoint`
- `ActivationCache`

So the v1 implementation does not wrap MusicGen with vanilla TransformerLens from scratch. Instead, it depends on the MusicGen-specific TransformerLens-style adapter from `musicdiscovery`.

## RunPod Setup

On a GPU pod:

```bash
cd /workspace
git clone https://github.com/PapayaResearch/musicdiscovery.git
cd musicdiscovery
pip install -r requirements.txt
```

Then return to this repo and install local dependencies:

```bash
cd /workspace/musicgen-exp
uv sync --extra dev --extra audio
```

## Extract Activations

After creating a real benchmark manifest and placing the referenced audio files locally:

```bash
uv run python scripts/extract_musicgen_activations.py \
  --config configs/experiment.yaml \
  --manifest data/benchmark_manifest.jsonl \
  --audio-root /path/to/mtg-jamendo-audio \
  --musicdiscovery-path /workspace/musicdiscovery \
  --model-size pilot \
  --device cuda \
  --output-dir outputs/activations
```

To add SAE feature activations, pass a checkpoint root:

```bash
uv run python scripts/extract_musicgen_activations.py \
  --manifest data/benchmark_manifest.jsonl \
  --audio-root /path/to/mtg-jamendo-audio \
  --musicdiscovery-path /workspace/musicdiscovery \
  --sae-checkpoint-root /path/to/musicdiscovery/checkpoints \
  --model-size primary \
  --device cuda
```

The command fails if real manifest/audio files, musicdiscovery, AudioCraft, TransformerLens, SAE Lens, or checkpoints are missing.

Teacher-forced extraction uses real audio as input and writes residual plus optional SAE feature activations:

```bash
uv run python scripts/extract_musicgen_activations.py \
  --mode teacher-forced \
  --manifest data/benchmark_manifest.jsonl \
  --audio-root /path/to/mtg-jamendo-audio \
  --musicdiscovery-path /workspace/musicdiscovery \
  --model-size pilot \
  --device cuda \
  --output-dir outputs/activations/teacher_forced
```

Generation-time extraction uses real prefixes, calls `HookedMusicGen.generate_continuation()`, and records hook activations during sampling:

```bash
uv run python scripts/extract_musicgen_activations.py \
  --mode generation \
  --manifest data/benchmark_manifest.jsonl \
  --audio-root /path/to/mtg-jamendo-audio \
  --musicdiscovery-path /workspace/musicdiscovery \
  --model-size pilot \
  --device cuda \
  --output-dir outputs/activations/generation
```

## Hook Naming

Residual stream hooks follow the upstream naming convention:

```text
hook_layers.<layer>
```

This mirrors the relevant TransformerLens mental model while staying faithful to MusicGen's architecture.
