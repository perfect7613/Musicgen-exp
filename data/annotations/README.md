# Benchmark Annotations

This directory is reserved for annotations derived from real MTG-Jamendo tracks and real MusicGen continuations.

Do not commit placeholder annotations, synthetic motif examples, generated toy labels, or fake model outputs. Final benchmark records must preserve real source metadata, license information, selected windows, and manual verification status.

Use schema-only validation before real annotations exist:

```bash
uv run python scripts/validate_annotations.py --schema-only
```
