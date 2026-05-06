# Research Plan

## Summary

This project extends Singh et al.'s MusicGen SAE work from concept discovery and steering into temporal causality. The core question is whether MusicGen contains features that help preserve or reintroduce musical motifs later in generation.

## Model And SAE Setup

- Primary model: MusicGen-Large.
- Pilot model: MusicGen-Small.
- Base implementation: PapayaResearch `musicdiscovery`.
- Priority SAE configuration from Singh et al.: MusicGen-Large, expansion factor 32, `k=100`.
- Priority layers: `L24`, `L36`, `L46`.
- Secondary layers if compute allows: `L2`, `L12`.

## Benchmark

- Source: MTG-Jamendo, filtered for instrumental Creative Commons tracks.
- Target: 100 clips.
- Clip format: 10s real-music prefix + 30s MusicGen continuation.
- Primary label: motif recurrence in the continuation.
- Annotation strategy: chroma/DTW proposals plus manual verification.
- Release strategy: track IDs, timestamps, licenses, annotations, and scripts. Do not rehost audio unless each license clearly permits it.
- Data policy: benchmark annotations and experiment artifacts must come from real MTG-Jamendo metadata/audio and real MusicGen/model execution; fake musical data and placeholder benchmark records are prohibited.

## Experiments

1. Reproduce activation extraction and SAE loading from `musicdiscovery`.
2. Build the MTG-Jamendo instrumental benchmark.
3. Extract teacher-forced activations from real/prefix audio.
4. Train future-event probes for motif recurrence at local, medium, and long horizons.
5. Select candidate SAE features on a train split.
6. Test candidate features on held-out clips.
7. Run generation-time SAE feature ablation/scaling.
8. Compare against random features, local-only features, position-only controls, acoustic controls, and shuffled labels.

## Go / No-Go Gates

- Gate 1: `musicdiscovery` reproduction works.
- Gate 2: SAE checkpoints load and produce sensible activations.
- Gate 3: Motif recurrence annotations are reliable after manual verification.
- Gate 4: Future-event probes beat controls on held-out data.
- Gate 5: Causal interventions produce recurrence-specific effects beyond controls.

If a gate fails, document the failure as a useful null or methods note.

## Implementation Spine

All runs should start by validating `configs/experiment.yaml` and creating a run manifest. See `docs/pipeline.md` for the current staged workflow.
