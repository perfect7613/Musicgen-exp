# Real-Data-Only Policy

This project is a mechanistic interpretability study, so evidence quality matters more than convenience. The implementation must not rely on fake musical data, synthetic motif fixtures, placeholder benchmark records, or fake model outputs.

## Allowed

- Real MTG-Jamendo metadata downloaded from the official dataset sources.
- Real audio files that the user has downloaded locally according to dataset/license rules.
- Real MusicGen continuations generated from real benchmark prefixes.
- Structural tests that validate schemas, config shapes, CLI behavior, and error handling without pretending to be music data.
- Small non-musical dictionaries or empty inputs used only to test validation failures.

## Not Allowed

- Synthetic audio clips or toy motifs used as benchmark examples.
- Placeholder annotation JSON that looks like a real benchmark item.
- Fake MusicGen outputs or fake SAE activations represented as experiment evidence.
- Tests that pass because invented musical labels or fake recurrence windows were accepted.
- Rehosting MTG-Jamendo source audio unless each license clearly permits it.

## Practical Rule

If a file claims to represent a track, clip, motif, recurrence window, activation artifact, probe result, or intervention output, it must come from real data/model execution and carry enough provenance to audit where it came from.
