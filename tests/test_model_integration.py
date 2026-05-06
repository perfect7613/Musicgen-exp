from __future__ import annotations

from pathlib import Path

import pytest

from musicgen_exp.model_integration import (
    ModelSpec,
    add_musicdiscovery_to_path,
    build_sae_specs,
    load_hooked_musicgen,
)


def test_musicdiscovery_path_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="musicdiscovery checkout"):
        add_musicdiscovery_to_path(tmp_path / "missing")


def test_missing_musicdiscovery_import_fails_cleanly() -> None:
    spec = ModelSpec(
        model_name="facebook/musicgen-small",
        device="cpu",
        precision="float32",
        layers=[2],
        musicdiscovery_path=None,
    )

    with pytest.raises(RuntimeError, match="HookedMusicGen"):
        load_hooked_musicgen(spec)


def test_sae_checkpoint_root_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="SAE checkpoint root"):
        build_sae_specs(tmp_path / "missing", [24])


def test_empty_sae_checkpoint_root_rejects_missing_layer(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="No SAE checkpoint candidate"):
        build_sae_specs(root, [24])
