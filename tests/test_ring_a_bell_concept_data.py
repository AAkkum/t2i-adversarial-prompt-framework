"""Validation for the final non-sexual Ring-A-Bell concept-pair file."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from t2i_framework.attacks.ring_a_bell_encoder import load_pairs

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/datasets/safety_nonsexual/safety_nonsexual_100.csv"
PAIR_PATH = ROOT / "data/ring_a_bell/concept_pairs_safety_nonsexual_100.json"


def _targets() -> set[str]:
    with DATASET_PATH.open(encoding="utf-8-sig", newline="") as handle:
        return {row["target_concept"] for row in csv.DictReader(handle)}


@pytest.mark.parametrize("target", sorted(_targets()))
def test_every_final_target_has_valid_pairs(target: str) -> None:
    pairs, digest = load_pairs(PAIR_PATH, target)

    assert len(pairs) >= 5
    assert digest == hashlib.sha256(PAIR_PATH.read_bytes()).hexdigest()
    assert len(set(pairs)) == len(pairs)
    assert all(positive != negative for positive, negative in pairs)


def test_pair_file_records_final_dataset_provenance() -> None:
    data = json.loads(PAIR_PATH.read_text(encoding="utf-8"))

    assert data["provenance"]["source_csv"] == (
        "data/datasets/safety_nonsexual/safety_nonsexual_100.csv"
    )
    assert data["provenance"]["author_data_included"] is False
    assert _targets() <= set(data["concepts"])


def test_lookup_normalizes_case_and_whitespace() -> None:
    assert load_pairs(PAIR_PATH, "MOLOTOV COCKTAIL") == load_pairs(
        PAIR_PATH, "  Molotov\t cocktail  "
    )


def test_unknown_concept_fails_clearly() -> None:
    with pytest.raises(ValueError, match="No positive/negative pairs configured"):
        load_pairs(PAIR_PATH, "not configured")


def test_normalized_collisions_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pairs.json"
    entries = [{"positive": "one", "negative": "two"}]
    path.write_text(json.dumps({"concepts": {"term": entries, " TERM ": entries}}))

    with pytest.raises(ValueError, match="Ambiguous concept keys"):
        load_pairs(path, "term")
