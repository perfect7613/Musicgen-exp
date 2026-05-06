# Evaluation And Dashboards

Evaluation consumes real probe results and real intervention outputs.

## Run

```bash
uv run python scripts/generate_evaluation_report.py \
  --probe-results outputs/probes/sae/probe_results.jsonl \
  --intervention-index outputs/interventions/intervention_index.jsonl \
  --output-dir outputs/evaluation
```

## Outputs

- `evaluation_summary.json`: machine-readable probe/intervention summary and go/no-go status.
- `feature_dashboard.md`: write-up-ready Markdown table with probe margins, intervention recurrence distances, quality flags, and audio demo paths.

## Interpretation

The dashboard is not itself evidence of foresight-like coherence. It is an audit surface. Claims should remain null/inconclusive unless:

- Long-horizon probes beat controls on held-out data.
- Ablation and scaling produce recurrence-specific changes.
- Quality flags do not indicate broad degradation.
