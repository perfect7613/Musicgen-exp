from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from musicgen_exp.audio_features import require_existing_file


CONTROL_SETS = ("main", "position_only", "acoustic_only", "prompt_source", "shuffled_labels")


@dataclass(frozen=True)
class ProbeResult:
    horizon_name: str
    control_name: str
    n_train: int
    n_eval: int
    auc: float | None
    f1: float | None
    calibration_brier: float | None
    candidate_feature_ranking_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_probe_suite(
    activation_index_path: str | Path,
    annotation_splits_path: str | Path,
    output_dir: str | Path,
    horizon_names: list[str],
    activation_kind: str = "residual",
) -> Path:
    activation_index = load_activation_index(activation_index_path)
    splits = load_annotation_splits(annotation_splits_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[ProbeResult] = []
    for horizon_name in horizon_names:
        train_rows = build_probe_rows(
            annotations=splits["splits"]["train"],
            activation_index=activation_index,
            activation_kind=activation_kind,
        )
        eval_rows = build_probe_rows(
            annotations=splits["splits"].get("validation", []) + splits["splits"].get("test", []),
            activation_index=activation_index,
            activation_kind=activation_kind,
        )
        for control_name in CONTROL_SETS:
            result = fit_and_evaluate_probe(
                train_rows=train_rows,
                eval_rows=eval_rows,
                output_root=output_root,
                horizon_name=horizon_name,
                control_name=control_name,
            )
            results.append(result)

    results_path = output_root / "probe_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for result in results:
            json.dump(result.to_dict(), f, sort_keys=True)
            f.write("\n")
    return results_path


def load_activation_index(path: str | Path) -> dict[str, dict[str, Any]]:
    index_path = require_existing_file(path, "activation index JSONL")
    rows: dict[str, dict[str, Any]] = {}
    with index_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{index_path}:{line_number}: invalid JSON") from exc
            track_id = str(row.get("track_id", ""))
            if not track_id:
                raise ValueError(f"{index_path}:{line_number}: activation row missing track_id")
            rows[track_id] = row
    return rows


def load_annotation_splits(path: str | Path) -> dict[str, Any]:
    splits_path = require_existing_file(path, "annotation splits JSON")
    with splits_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict) or "splits" not in loaded:
        raise ValueError(f"{splits_path}: expected annotation split object")
    return loaded


def build_probe_rows(
    annotations: list[dict[str, Any]],
    activation_index: dict[str, dict[str, Any]],
    activation_kind: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for annotation in annotations:
        track_id = str(annotation["track_id"])
        activation_row = activation_index.get(track_id)
        if activation_row is None:
            raise ValueError(f"{track_id}: no activation artifact found")
        vector = load_activation_vector(activation_row, activation_kind)
        rows.append(
            {
                "track_id": track_id,
                "label": label_from_annotation(annotation),
                "activation": vector,
                "position": position_features(annotation),
                "acoustic": acoustic_features(annotation),
                "prompt_source": prompt_source_features(annotation),
            }
        )
    return rows


def load_activation_vector(activation_row: dict[str, Any], activation_kind: str) -> np.ndarray:
    key = "sae_paths" if activation_kind == "sae" else "residual_paths"
    paths = activation_row.get(key)
    if not isinstance(paths, dict) or not paths:
        raise ValueError(f"{activation_row['track_id']}: no {activation_kind} activation paths found")
    vectors: list[np.ndarray] = []
    for hook_name in sorted(paths):
        path = require_existing_file(paths[hook_name], f"{activation_kind} activation artifact")
        activation = np.load(path)
        vectors.append(summarize_activation(activation))
    return np.concatenate(vectors)


def summarize_activation(activation: np.ndarray) -> np.ndarray:
    if activation.size == 0:
        raise ValueError("activation artifact is empty")
    flattened = activation.reshape(-1, activation.shape[-1])
    return np.concatenate([flattened.mean(axis=0), flattened.std(axis=0)])


def label_from_annotation(annotation: dict[str, Any]) -> int:
    status = annotation.get("manual_verification_status")
    event_type = annotation.get("event_type")
    if status == "verified_positive" or event_type == "motif_recurrence":
        return 1
    if status == "verified_negative" or event_type == "no_recurrence":
        return 0
    raise ValueError(f"{annotation.get('track_id', '<unknown>')}: annotation is not binary")


def position_features(annotation: dict[str, Any]) -> np.ndarray:
    motif = annotation["motif_window"]
    recurrence = annotation["recurrence_window"]
    return np.array(
        [
            float(motif["start_seconds"]),
            float(motif["end_seconds"]),
            float(recurrence["start_seconds"]),
            float(recurrence["end_seconds"]),
        ],
        dtype=float,
    )


def acoustic_features(annotation: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(annotation.get("energy", 0.0)),
            float(annotation.get("spectral_centroid", 0.0)),
        ],
        dtype=float,
    )


def prompt_source_features(annotation: dict[str, Any]) -> np.ndarray:
    source = str(annotation.get("source_dataset", ""))
    prompt = str(annotation.get("prompt", ""))
    return np.array(
        [
            stable_hash_fraction(source),
            stable_hash_fraction(prompt),
        ],
        dtype=float,
    )


def fit_and_evaluate_probe(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    output_root: Path,
    horizon_name: str,
    control_name: str,
) -> ProbeResult:
    if not train_rows or not eval_rows:
        raise ValueError("probe training requires non-empty train and eval rows")
    x_train, y_train = matrix_for_control(train_rows, control_name, shuffle_labels=False)
    x_eval, y_eval = matrix_for_control(eval_rows, control_name, shuffle_labels=False)
    if control_name == "shuffled_labels":
        _, y_train = matrix_for_control(train_rows, control_name, shuffle_labels=True)

    model = fit_logistic_regression(x_train, y_train)
    probabilities = model.predict_proba(x_eval)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    ranking_path = None
    if control_name == "main":
        ranking_path = str(
            write_candidate_feature_ranking(
                model.named_steps["logisticregression"].coef_[0],
                output_root,
                horizon_name,
            )
        )

    return ProbeResult(
        horizon_name=horizon_name,
        control_name=control_name,
        n_train=len(train_rows),
        n_eval=len(eval_rows),
        auc=safe_auc(y_eval, probabilities),
        f1=safe_f1(y_eval, predictions),
        calibration_brier=safe_brier(y_eval, probabilities),
        candidate_feature_ranking_path=ranking_path,
    )


def matrix_for_control(
    rows: list[dict[str, Any]],
    control_name: str,
    shuffle_labels: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if control_name in {"main", "shuffled_labels"}:
        matrix = np.stack([row["activation"] for row in rows])
    elif control_name == "position_only":
        matrix = np.stack([row["position"] for row in rows])
    elif control_name == "acoustic_only":
        matrix = np.stack([row["acoustic"] for row in rows])
    elif control_name == "prompt_source":
        matrix = np.stack([row["prompt_source"] for row in rows])
    else:
        raise ValueError(f"unknown control set: {control_name}")
    labels = np.array([row["label"] for row in rows], dtype=int)
    if shuffle_labels:
        labels = deterministic_label_shuffle(labels)
    return matrix, labels


def fit_logistic_regression(x_train: np.ndarray, y_train: np.ndarray) -> Any:
    if len(set(y_train.tolist())) < 2:
        raise ValueError("probe training labels must contain both classes")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError("install analysis dependencies with `uv sync --extra analysis`") from exc
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    ).fit(x_train, y_train)


def safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, probabilities))


def safe_f1(y_true: np.ndarray, predictions: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, predictions))


def safe_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    from sklearn.metrics import brier_score_loss

    return float(brier_score_loss(y_true, probabilities))


def write_candidate_feature_ranking(coefs: np.ndarray, output_root: Path, horizon_name: str) -> Path:
    ranking_path = output_root / f"{horizon_name}.candidate_feature_ranking.jsonl"
    order = np.argsort(np.abs(coefs))[::-1]
    with ranking_path.open("w", encoding="utf-8") as f:
        for rank, feature_index in enumerate(order, start=1):
            json.dump(
                {
                    "rank": rank,
                    "feature_index": int(feature_index),
                    "coefficient": float(coefs[feature_index]),
                    "abs_coefficient": float(abs(coefs[feature_index])),
                },
                f,
                sort_keys=True,
            )
            f.write("\n")
    return ranking_path


def deterministic_label_shuffle(labels: np.ndarray) -> np.ndarray:
    shuffled = labels.copy()
    rng = np.random.default_rng(7613)
    rng.shuffle(shuffled)
    return shuffled


def stable_hash_fraction(value: str) -> float:
    import hashlib

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF
