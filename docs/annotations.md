# Manual Annotation Workflow

Recurrence proposals are not final labels. They must be manually reviewed before use in probes or causal tests.

## Prepare Review Queue

```bash
uv run python scripts/prepare_annotation_review.py \
  --manifest data/benchmark_manifest.jsonl \
  --proposals data/recurrence_proposals.jsonl \
  --output data/annotations/review_queue.jsonl
```

The output rows start as `manual_verification_status: unverified`.

## Manual Statuses

Allowed final statuses:

- `verified_positive`: the proposed recurrence is a real motif recurrence.
- `verified_negative`: the proposed recurrence is not a motif recurrence.
- `ambiguous`: the example is too uncertain for a clean binary label.
- `rejected`: the item should not be used.

Unverified rows must not enter final experiment splits.

## Create Splits

After manual review:

```bash
uv run python scripts/create_annotation_splits.py \
  --annotations data/annotations/verified.jsonl \
  --schema schemas/annotation.schema.json \
  --output data/annotation_splits.json
```

Splits are deterministic from `track_id`, which prevents accidental reshuffling between runs.
