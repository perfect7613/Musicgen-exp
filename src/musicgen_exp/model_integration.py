from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from musicgen_exp.audio_features import load_manifest_jsonl, require_existing_audio


HOOK_NAME_TEMPLATE = "hook_layers.{layer}"


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    device: str
    precision: str
    layers: list[int]
    musicdiscovery_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SAECheckpointSpec:
    layer: int
    checkpoint_path: str
    hook_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def add_musicdiscovery_to_path(path: str | Path | None) -> None:
    if path is None:
        return
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_dir():
        raise FileNotFoundError(f"musicdiscovery checkout not found: {resolved}")
    sys.path.insert(0, str(resolved))


def load_hooked_musicgen(spec: ModelSpec) -> Any:
    add_musicdiscovery_to_path(spec.musicdiscovery_path)
    try:
        from sae_components.musicgen_hooked import HookedMusicGen
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import musicdiscovery HookedMusicGen. Clone "
            "https://github.com/PapayaResearch/musicdiscovery and install its RunPod "
            "dependencies, then pass --musicdiscovery-path."
        ) from exc
    return HookedMusicGen(spec.model_name, device=spec.device)


def load_sae_checkpoint(spec: SAECheckpointSpec, device: str) -> Any:
    checkpoint_path = Path(spec.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found for layer {spec.layer}: {checkpoint_path}")
    try:
        from sae_lens import SAE
    except ModuleNotFoundError as exc:
        raise RuntimeError("Could not import sae_lens. Install musicdiscovery dependencies.") from exc

    if hasattr(SAE, "load_from_pretrained"):
        return SAE.load_from_pretrained(str(checkpoint_path), device=device)
    if hasattr(SAE, "from_pretrained"):
        return SAE.from_pretrained(str(checkpoint_path), device=device)
    raise RuntimeError("Installed sae_lens.SAE has no supported checkpoint loading method")


def build_sae_specs(checkpoint_root: str | Path | None, layers: list[int]) -> list[SAECheckpointSpec]:
    if checkpoint_root is None:
        return []
    root = Path(checkpoint_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"SAE checkpoint root not found: {root}")
    specs: list[SAECheckpointSpec] = []
    for layer in layers:
        candidates = sorted(root.glob(f"**/*L{layer}*")) + sorted(root.glob(f"**/*layer{layer}*"))
        if not candidates:
            raise FileNotFoundError(f"No SAE checkpoint candidate found for layer {layer} under {root}")
        specs.append(
            SAECheckpointSpec(
                layer=layer,
                checkpoint_path=str(candidates[0]),
                hook_name=HOOK_NAME_TEMPLATE.format(layer=layer),
            )
        )
    return specs


def extract_real_audio_activations(
    manifest_path: str | Path,
    audio_root: str | Path,
    output_dir: str | Path,
    model_spec: ModelSpec,
    sae_specs: list[SAECheckpointSpec] | None = None,
    limit: int | None = None,
) -> Path:
    manifest = load_manifest_jsonl(manifest_path)
    selected_items = manifest[:limit] if limit is not None else manifest
    torch = import_torch()
    librosa = import_librosa()
    model = load_hooked_musicgen(model_spec)
    sae_by_layer = {
        spec.layer: load_sae_checkpoint(spec, model_spec.device) for spec in (sae_specs or [])
    }

    artifact_root = Path(output_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    for item in selected_items:
        track_id = str(item["track_id"])
        audio_path = Path(audio_root) / str(item["audio_path"])
        require_existing_audio(audio_path)
        audio, sample_rate = librosa.load(audio_path, sr=32000, mono=True)
        audio_tensor = torch.tensor(audio, dtype=torch.float32, device=model_spec.device).unsqueeze(0)

        with torch.no_grad():
            _, cache = model.run_with_cache(audio_tensor, return_cache_object=False)

        track_output_dir = artifact_root / track_id
        track_output_dir.mkdir(parents=True, exist_ok=True)
        residual_paths: dict[str, str] = {}
        sae_paths: dict[str, str] = {}
        for layer in model_spec.layers:
            hook_name = HOOK_NAME_TEMPLATE.format(layer=layer)
            if hook_name not in cache:
                raise KeyError(f"{track_id}: hook {hook_name} not found in MusicGen cache")
            activation = cache[hook_name].detach().cpu().numpy()
            residual_path = track_output_dir / f"{hook_name}.residual.npy"
            np.save(residual_path, activation)
            residual_paths[hook_name] = str(residual_path)

            sae = sae_by_layer.get(layer)
            if sae is not None:
                sae_activation = encode_sae_features(sae, cache[hook_name]).detach().cpu().numpy()
                sae_path = track_output_dir / f"{hook_name}.sae.npy"
                np.save(sae_path, sae_activation)
                sae_paths[hook_name] = str(sae_path)

        index_rows.append(
            {
                "track_id": track_id,
                "audio_path": str(audio_path),
                "sample_rate": int(sample_rate),
                "model": model_spec.to_dict(),
                "residual_paths": residual_paths,
                "sae_paths": sae_paths,
            }
        )

    index_path = artifact_root / "activation_index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for row in index_rows:
            json.dump(row, f, sort_keys=True)
            f.write("\n")
    return index_path


def extract_real_generation_activations(
    manifest_path: str | Path,
    audio_root: str | Path,
    output_dir: str | Path,
    model_spec: ModelSpec,
    limit: int | None = None,
) -> Path:
    manifest = load_manifest_jsonl(manifest_path)
    selected_items = manifest[:limit] if limit is not None else manifest
    torch = import_torch()
    librosa = import_librosa()
    model = load_hooked_musicgen(model_spec)

    artifact_root = Path(output_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    for item in selected_items:
        track_id = str(item["track_id"])
        audio_path = Path(audio_root) / str(item["audio_path"])
        require_existing_audio(audio_path)
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

        captured: dict[str, list[np.ndarray]] = {
            HOOK_NAME_TEMPLATE.format(layer=layer): [] for layer in model_spec.layers
        }

        def save_activation(value: Any, hook: Any) -> Any:
            captured[hook.name].append(value.detach().cpu().numpy())
            return value

        hooks = [
            (HOOK_NAME_TEMPLATE.format(layer=layer), save_activation) for layer in model_spec.layers
        ]
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                generated_audio = model.generate_continuation(
                    prompt=prompt_tensor,
                    prompt_sample_rate=sample_rate,
                    descriptions=[str(item.get("prompt", ""))],
                    progress=False,
                )

        track_output_dir = artifact_root / track_id
        track_output_dir.mkdir(parents=True, exist_ok=True)
        generated_audio_path = track_output_dir / "generated_continuation.npy"
        np.save(generated_audio_path, generated_audio.detach().cpu().numpy())

        generation_paths: dict[str, str] = {}
        for hook_name, activation_calls in captured.items():
            if not activation_calls:
                raise KeyError(f"{track_id}: hook {hook_name} captured no generation activations")
            generation_path = track_output_dir / f"{hook_name}.generation.npz"
            np.savez_compressed(
                generation_path,
                **{f"call_{index:05d}": value for index, value in enumerate(activation_calls)},
            )
            generation_paths[hook_name] = str(generation_path)

        index_rows.append(
            {
                "track_id": track_id,
                "audio_path": str(audio_path),
                "sample_rate": int(sample_rate),
                "model": model_spec.to_dict(),
                "generated_audio_path": str(generated_audio_path),
                "generation_activation_paths": generation_paths,
            }
        )

    index_path = artifact_root / "generation_activation_index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for row in index_rows:
            json.dump(row, f, sort_keys=True)
            f.write("\n")
    return index_path


def encode_sae_features(sae: Any, activation: Any) -> Any:
    if hasattr(sae, "encode"):
        return sae.encode(activation)
    if hasattr(sae, "encode_standard"):
        return sae.encode_standard(activation)
    if callable(sae):
        return sae(activation)
    raise RuntimeError("Loaded SAE object has no supported encoding interface")


def import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("install model dependencies on RunPod before activation extraction") from exc
    return torch


def import_librosa() -> Any:
    try:
        import librosa
    except ModuleNotFoundError as exc:
        raise RuntimeError("install audio dependencies with `uv sync --extra audio --extra dev`") from exc
    return librosa
