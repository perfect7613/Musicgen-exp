# Benchmark Manifest

The benchmark manifest must be built from real MTG-Jamendo metadata and per-track license information. Do not create placeholder manifest rows.

## Required Inputs

Download or clone the official MTG-Jamendo dataset metadata repository:

```bash
git clone https://github.com/MTG/mtg-jamendo-dataset.git external/mtg-jamendo-dataset
```

The first manifest builder expects:

- `data/autotagging_instrument.tsv` or another official MTG-Jamendo TSV with the same structure.
- `audio_licenses.txt`.

The dataset documentation states that MTG-Jamendo contains Creative Commons audio with individual licenses in `audio_licenses.txt`, and that metadata/audio use has research/non-commercial restrictions. Preserve license provenance in every benchmark artifact.

## Build

```bash
uv run python scripts/build_benchmark_manifest.py \
  --metadata-tsv external/mtg-jamendo-dataset/data/autotagging_instrument.tsv \
  --audio-licenses external/mtg-jamendo-dataset/audio_licenses.txt \
  --output data/benchmark_manifest.jsonl
```

The generated manifest uses:

- A 10-second prefix window.
- A 30-second continuation window.
- Tracks with enough duration for the full window.
- Tracks with instrument tags.
- Exclusion of voice/vocal tags for the v1 instrumental benchmark.

Source audio is not rehosted by this repository.
