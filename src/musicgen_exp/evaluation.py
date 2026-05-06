from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from musicgen_exp.audio_features import require_existing_file


@dataclass(frozen=True)
class QualityThresholds:
    max_loudness_ratio: float = 2.5
    max_recurrence_distance_ratio: float = 1.5


def generate_evaluation_report(
    probe_results_path: str | Path,
    intervention_index_path: str | Path,
    output_dir: str | Path,
    thresholds: QualityThresholds | None = None,
) -> Path:
    thresholds = thresholds or QualityThresholds()
    probe_rows = load_jsonl(probe_results_path, "probe results JSONL")
    intervention_rows = load_jsonl(intervention_index_path, "intervention index JSONL")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    probe_summary = summarize_probe_results(probe_rows)
    intervention_summary = summarize_interventions(intervention_rows, thresholds)

    summary_path = output_root / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "probe_summary": probe_summary,
                "intervention_summary": intervention_summary,
                "go_no_go": go_no_go_summary(probe_summary, intervention_summary),
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    dashboard_path = output_root / "feature_dashboard.md"
    dashboard_path.write_text(
        render_dashboard_markdown(probe_summary, intervention_summary),
        encoding="utf-8",
    )
    return dashboard_path


def load_jsonl(path: str | Path, description: str) -> list[dict[str, Any]]:
    input_path = require_existing_file(path, description)
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{line_number}: expected JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{input_path}: no rows found")
    return rows


def summarize_probe_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_horizon: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_horizon.setdefault(str(row["horizon_name"]), []).append(row)

    summary: dict[str, Any] = {}
    for horizon_name, horizon_rows in sorted(by_horizon.items()):
        main = find_control(horizon_rows, "main")
        controls = [row for row in horizon_rows if row.get("control_name") != "main"]
        best_control_auc = max(
            [float(row["auc"]) for row in controls if row.get("auc") is not None],
            default=None,
        )
        main_auc = None if main.get("auc") is None else float(main["auc"])
        summary[horizon_name] = {
            "main_auc": main_auc,
            "main_f1": main.get("f1"),
            "main_calibration_brier": main.get("calibration_brier"),
            "best_control_auc": best_control_auc,
            "auc_margin_over_best_control": (
                None
                if main_auc is None or best_control_auc is None
                else float(main_auc - best_control_auc)
            ),
            "candidate_feature_ranking_path": main.get("candidate_feature_ranking_path"),
        }
    return summary


def summarize_interventions(
    rows: list[dict[str, Any]],
    thresholds: QualityThresholds,
) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        condition = str(row["intervention"]["condition"])
        by_condition.setdefault(condition, []).append(row)

    baseline_distances = [
        float(row["recurrence_distance"])
        for row in by_condition.get("baseline", [])
        if row.get("recurrence_distance") is not None
    ]
    baseline_mean = float(np.mean(baseline_distances)) if baseline_distances else None
    summary: dict[str, Any] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        distances = [
            float(row["recurrence_distance"])
            for row in condition_rows
            if row.get("recurrence_distance") is not None
        ]
        generated_paths = [str(row["generated_audio_path"]) for row in condition_rows]
        mean_distance = float(np.mean(distances)) if distances else None
        summary[condition] = {
            "count": len(condition_rows),
            "mean_recurrence_distance": mean_distance,
            "mean_distance_delta_vs_baseline": (
                None
                if baseline_mean is None or mean_distance is None
                else float(mean_distance - baseline_mean)
            ),
            "generated_audio_paths": generated_paths,
            "quality_flags": quality_flags(
                condition=condition,
                mean_distance=mean_distance,
                baseline_mean=baseline_mean,
                thresholds=thresholds,
            ),
        }
    return summary


def find_control(rows: list[dict[str, Any]], control_name: str) -> dict[str, Any]:
    for row in rows:
        if row.get("control_name") == control_name:
            return row
    raise ValueError(f"missing {control_name} probe row")


def quality_flags(
    condition: str,
    mean_distance: float | None,
    baseline_mean: float | None,
    thresholds: QualityThresholds,
) -> list[str]:
    flags: list[str] = []
    if condition != "baseline" and baseline_mean and mean_distance:
        ratio = mean_distance / baseline_mean
        if ratio > thresholds.max_recurrence_distance_ratio:
            flags.append("recurrence_distance_degraded_vs_baseline")
    return flags


def go_no_go_summary(
    probe_summary: dict[str, Any],
    intervention_summary: dict[str, Any],
) -> dict[str, Any]:
    long_horizon = probe_summary.get("long_seconds", {})
    ablate = intervention_summary.get("ablate", {})
    scale = intervention_summary.get("scale", {})
    probe_pass = (long_horizon.get("auc_margin_over_best_control") or 0.0) > 0.05
    intervention_pass = (
        ablate.get("mean_distance_delta_vs_baseline") is not None
        and scale.get("mean_distance_delta_vs_baseline") is not None
    )
    return {
        "probe_gate_passed": bool(probe_pass),
        "intervention_gate_has_signal": bool(intervention_pass),
        "recommended_claim": (
            "candidate_long_horizon_coherence_features"
            if probe_pass and intervention_pass
            else "null_or_inconclusive"
        ),
    }


def render_dashboard_markdown(
    probe_summary: dict[str, Any],
    intervention_summary: dict[str, Any],
) -> str:
    lines = [
        "# Feature Dashboard",
        "",
        "This dashboard summarizes real probe and intervention artifacts. It should not be edited into a positive claim unless the go/no-go gates support it.",
        "",
        "## Probe Summary",
        "",
        "| Horizon | Main AUC | Best Control AUC | Margin | Candidate Ranking |",
        "|---|---:|---:|---:|---|",
    ]
    for horizon, row in sorted(probe_summary.items()):
        lines.append(
            "| {horizon} | {main_auc} | {control_auc} | {margin} | {ranking} |".format(
                horizon=horizon,
                main_auc=format_optional_float(row.get("main_auc")),
                control_auc=format_optional_float(row.get("best_control_auc")),
                margin=format_optional_float(row.get("auc_margin_over_best_control")),
                ranking=row.get("candidate_feature_ranking_path") or "",
            )
        )
    lines.extend(
        [
            "",
            "## Intervention Summary",
            "",
            "| Condition | Count | Mean Recurrence Distance | Delta vs Baseline | Quality Flags |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for condition, row in sorted(intervention_summary.items()):
        lines.append(
            "| {condition} | {count} | {distance} | {delta} | {flags} |".format(
                condition=condition,
                count=row["count"],
                distance=format_optional_float(row.get("mean_recurrence_distance")),
                delta=format_optional_float(row.get("mean_distance_delta_vs_baseline")),
                flags=", ".join(row["quality_flags"]),
            )
        )
    lines.extend(
        [
            "",
            "## Audio Demo Index",
            "",
        ]
    )
    for condition, row in sorted(intervention_summary.items()):
        lines.append(f"### {condition}")
        for path in row["generated_audio_paths"]:
            lines.append(f"- `{path}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def format_optional_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"
