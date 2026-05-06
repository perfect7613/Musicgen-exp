from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from musicgen_exp.audio_features import load_manifest_jsonl, require_existing_file
from musicgen_exp.model_integration import (
    HOOK_NAME_TEMPLATE,
    ModelSpec,
    build_sae_specs,
    encode_sae_features,
    import_librosa,
    import_torch,
    load_hooked_musicgen,
    load_sae_checkpoint,
)


CONDITIONS = ("baseline", "ablate", "scale", "random_feature", "local_feature")


@dataclass(frozen=True)
class InterventionSpec:
    condition: str
    layer: int
    feature_index: int | None
    strength: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_intervention_suite(
    manifest_path: str | Path,
    candidate_ranking_path: str | Path,
    audio_root: str | Path,
    output_dir: str | Path,
    model_spec: ModelSpec,
    sae_checkpoint_root: str | Path,
    strength: float,
    seeds: list[int],
    top_k: int,
    layer_override: int | None = None,
    limit: int | None = None,
) -> Path:
    manifest = load_manifest_jsonl(manifest_path)
    candidates = load_candidate_features(
        candidate_ranking_path,
        top_k=top_k,
        layer_override=layer_override,
    )
    selected_items = manifest[:limit] if limit is not None else manifest
    torch = import_torch()
    librosa = import_librosa()
    model = load_hooked_musicgen(model_spec)
    sae_specs = build_sae_specs(sae_checkpoint_root, model_spec.layers)
    sae_by_layer = {spec.layer: load_sae_checkpoint(spec, model_spec.device) for spec in sae_specs}

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in selected_items:
        track_id = str(item["track_id"])
        audio_path = Path(audio_root) / str(item["audio_path"])
        require_existing_file(audio_path, "real benchmark audio")
        prefix_window = item["prefix_window"]
        generation_window = item["generation_window"]
        prefix_start = float(prefix_window["start_seconds"])
        prefix_duration = float(prefix_window["end_seconds"]) - prefix_start
        total_duration = float(generation_window["end_seconds"])
        prompt_audio, sample_rate = librosa.load(
            audio_path,
            sr=32000,
            mono=True,
            offset=prefix_start,
            duration=prefix_duration,
        )
        prompt_tensor = torch.tensor(prompt_audio, dtype=torch.float32, device=model_spec.device).unsqueeze(0)
        if hasattr(model.model, "set_generation_params"):
            model.model.set_generation_params(duration=total_duration)

        for seed in seeds:
            set_torch_seed(torch, seed)
            for candidate in candidates:
                layer = int(candidate["layer"])
                sae = sae_by_layer[layer]
                for condition in CONDITIONS:
                    feature_index = feature_for_condition(candidate, condition)
                    spec = InterventionSpec(
                        condition=condition,
                        layer=layer,
                        feature_index=feature_index,
                        strength=strength,
                        seed=seed,
                    )
                    generated_path = run_single_intervention(
                        model=model,
                        sae=sae,
                        prompt_tensor=prompt_tensor,
                        sample_rate=sample_rate,
                        track_output_dir=output_root / track_id,
                        spec=spec,
                    )
                    recurrence_distance = recurrence_distance_for_generated_audio(
                        prompt_audio=prompt_audio,
                        generated_audio_path=generated_path,
                        sample_rate=sample_rate,
                    )
                    rows.append(
                        {
                            "track_id": track_id,
                            "source_audio_path": str(audio_path),
                            "generated_audio_path": str(generated_path),
                            "recurrence_distance": recurrence_distance,
                            "intervention": spec.to_dict(),
                            "candidate": candidate,
                        }
                    )

    index_path = output_root / "intervention_index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, sort_keys=True)
            f.write("\n")
    write_recurrence_change_summary(rows, output_root / "recurrence_change_summary.json")
    return index_path


def run_single_intervention(
    model: Any,
    sae: Any,
    prompt_tensor: Any,
    sample_rate: int,
    track_output_dir: Path,
    spec: InterventionSpec,
) -> Path:
    torch = import_torch()
    track_output_dir.mkdir(parents=True, exist_ok=True)
    hook_name = HOOK_NAME_TEMPLATE.format(layer=spec.layer)
    fwd_hooks = []
    if spec.condition != "baseline":
        fwd_hooks = [(hook_name, build_intervention_hook(sae, spec))]
    with torch.no_grad():
        with model.hooks(fwd_hooks=fwd_hooks):
            generated_audio = model.generate_continuation(
                prompt=prompt_tensor,
                prompt_sample_rate=sample_rate,
                descriptions=[None],
                progress=False,
            )
    feature_label = "none" if spec.feature_index is None else str(spec.feature_index)
    path = (
        track_output_dir
        / f"{spec.condition}.L{spec.layer}.F{feature_label}.S{spec.seed}.generated.npy"
    )
    np.save(path, generated_audio.detach().cpu().numpy())
    return path


def build_intervention_hook(sae: Any, spec: InterventionSpec) -> Any:
    def hook_fn(value: Any, hook: Any) -> Any:
        if spec.feature_index is None:
            return value
        feature_acts = encode_sae_features(sae, value)
        decoder_vector = get_decoder_vector(sae, spec.feature_index, value.device)
        feature_values = feature_acts[..., spec.feature_index].unsqueeze(-1)
        delta = feature_values * decoder_vector
        if spec.condition == "ablate":
            return value - delta
        return value + (spec.strength * delta)

    return hook_fn


def get_decoder_vector(sae: Any, feature_index: int, device: Any) -> Any:
    if hasattr(sae, "W_dec"):
        return sae.W_dec[feature_index].to(device)
    if hasattr(sae, "decoder") and hasattr(sae.decoder, "weight"):
        return sae.decoder.weight[:, feature_index].to(device)
    raise RuntimeError("Loaded SAE has no supported decoder-vector interface")


def load_candidate_features(
    path: str | Path,
    top_k: int,
    layer_override: int | None = None,
) -> list[dict[str, Any]]:
    ranking_path = require_existing_file(path, "candidate feature ranking JSONL")
    candidates: list[dict[str, Any]] = []
    with ranking_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{ranking_path}:{line_number}: invalid JSON") from exc
            if "feature_index" not in row:
                raise ValueError(f"{ranking_path}:{line_number}: missing feature_index")
            layer = row.get("layer", layer_override)
            if layer is None:
                raise ValueError(
                    f"{ranking_path}:{line_number}: missing layer; pass --layer or use layer-aware rankings"
                )
            row["layer"] = int(layer)
            candidates.append(row)
            if len(candidates) >= top_k:
                break
    if not candidates:
        raise ValueError(f"{ranking_path}: no candidate features found")
    return candidates


def feature_for_condition(candidate: dict[str, Any], condition: str) -> int | None:
    if condition == "baseline":
        return None
    if condition == "random_feature":
        return int(candidate.get("random_control_feature_index", candidate["feature_index"]))
    if condition == "local_feature":
        return int(candidate.get("local_control_feature_index", candidate["feature_index"]))
    return int(candidate["feature_index"])


def set_torch_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_recurrence_change_summary(rows: list[dict[str, Any]], path: str | Path) -> Path:
    summary: dict[str, Any] = {"condition_counts": {}, "mean_recurrence_distance": {}}
    distances_by_condition: dict[str, list[float]] = {}
    for row in rows:
        condition = row["intervention"]["condition"]
        summary["condition_counts"][condition] = summary["condition_counts"].get(condition, 0) + 1
        distances_by_condition.setdefault(condition, []).append(float(row["recurrence_distance"]))
    for condition, distances in distances_by_condition.items():
        summary["mean_recurrence_distance"][condition] = float(np.mean(distances))
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return output_path


def recurrence_distance_for_generated_audio(
    prompt_audio: np.ndarray,
    generated_audio_path: str | Path,
    sample_rate: int,
    hop_length: int = 512,
) -> float:
    librosa = import_librosa()
    generated = np.load(generated_audio_path)
    generated = np.asarray(generated).reshape(-1)
    motif_chroma = librosa.feature.chroma_cqt(y=prompt_audio, sr=sample_rate, hop_length=hop_length)
    generated_chroma = librosa.feature.chroma_cqt(y=generated, sr=sample_rate, hop_length=hop_length)
    if motif_chroma.shape[1] < 2 or generated_chroma.shape[1] < 2:
        raise ValueError("not enough audio frames for recurrence DTW summary")
    distance = librosa.sequence.dtw(X=motif_chroma, Y=generated_chroma, metric="cosine")[0][-1, -1]
    return float(distance)
