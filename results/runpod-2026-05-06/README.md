---
title: MusicGen Long-Horizon Coherence RunPod Results
hf_dataset: Perfect7613/musicgen-exp-runpod-results
status: partial_real_run
date: 2026-05-06
---

# MusicGen Long-Horizon Coherence RunPod Results

This folder tracks the first real RunPod execution of the MusicGen long-horizon coherence experiment. Heavy tensors are stored in the private Hugging Face dataset repo `Perfect7613/musicgen-exp-runpod-results`; GitHub keeps the code, manifests, summaries, and findings.

## Completed Artifacts

- Source data: official MTG-Jamendo metadata plus `autotagging_moodtheme_audio-low-00.tar`.
- Data validation: tar SHA256 matched the official MTG-Jamendo manifest; 202 unpacked MP3 files were individually verified against official track SHA256 hashes.
- Benchmark manifest: 100 real instrument-labeled tracks selected only when the corresponding verified audio file existed.
- Model: `facebook/musicgen-small` via PapayaResearch/musicdiscovery `HookedMusicGen`.
- Activation extraction: completed for 100 tracks, producing 500 residual `.npy` tensors across five hooks.
- Captured hooks: `hook_layers.2`, `hook_layers.6`, `hook_layers.12`, `hook_layers.18`, `hook_layers.22`.
- Recurrence extraction: completed for 100 tracks, producing 100 chroma feature artifacts, 98 recurrence proposal rows, and 2 logged failures.
- Published SAE checkpoints: Singh et al. MusicGen-small 32x/k100 checkpoints were downloaded for hook layers `1`, `5`, `11`, `17`, and `21`.

## Important Caveats

- The residual activation run used pilot layers `2,6,12,18,22`, while the published SAE checkpoints use `1,5,11,17,21`. These should not be mixed; aligned SAE extraction requires rerunning residual extraction on the SAE checkpoint hook layers.
- Recurrence proposals are automatic chroma/DTW candidates, not manual labels. They are useful for review queues and exploratory analysis, but they are not acceptable as verified probe labels.
- No raw MTG-Jamendo audio is committed to GitHub or intended for the HF artifact repo.

## HF Artifact Layout

- `activations_full/`: full residual activation dump, expected 100 index rows and 500 residual tensor files.
- `features/`: 100 real-audio chroma feature artifacts.
- `metadata/`: benchmark manifest, activation index, logs, recurrence outputs, and checkpoint metadata.
- `sae_checkpoints/`: real Singh et al. MusicGen-small SAE checkpoint files.

## Links

- HF dataset: https://huggingface.co/datasets/Perfect7613/musicgen-exp-runpod-results
- Activation dump: https://huggingface.co/datasets/Perfect7613/musicgen-exp-runpod-results/tree/main/activations_full
- Metadata: https://huggingface.co/datasets/Perfect7613/musicgen-exp-runpod-results/tree/main/metadata
- Features: https://huggingface.co/datasets/Perfect7613/musicgen-exp-runpod-results/tree/main/features
- SAE checkpoints: https://huggingface.co/datasets/Perfect7613/musicgen-exp-runpod-results/tree/main/sae_checkpoints
