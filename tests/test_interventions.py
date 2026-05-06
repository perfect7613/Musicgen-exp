from __future__ import annotations

from pathlib import Path

import pytest

from musicgen_exp.interventions import load_candidate_features


def test_interventions_require_real_candidate_ranking(tmp_path: Path) -> None:
    missing_path = tmp_path / "ranking.jsonl"

    with pytest.raises(FileNotFoundError, match="candidate feature ranking JSONL"):
        load_candidate_features(missing_path, top_k=5)
