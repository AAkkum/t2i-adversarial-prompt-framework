from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data/datasets/safety_nonsexual"
MAIN_CSV = DATASET_DIR / "safety_nonsexual_100.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_main_dataset_structure_and_balance() -> None:
    rows = read_rows(MAIN_CSV)

    assert len(rows) == 100
    assert len({row["id"] for row in rows}) == 100
    assert len({row["prompt"] for row in rows}) == 100
    assert Counter(row["category"] for row in rows) == {
        "violence": 20,
        "illegal_activity": 20,
        "self_harm": 15,
        "shocking": 15,
        "hate": 15,
        "harassment": 15,
    }
    assert set(Counter(row["target_concept"] for row in rows).values()) == {5}
    assert len({row["target_concept"] for row in rows}) == 20


def test_prompts_are_explicit_nonsexual_descriptions() -> None:
    rows = read_rows(MAIN_CSV)
    forbidden = re.compile(
        r"\b(?:sexual|nude|nudity|naked|porn(?:ographic)?|erotic|fetish|genitals?)\b",
        re.IGNORECASE,
    )

    for row in rows:
        assert row["target_concept"].casefold() in row["prompt"].casefold()
        assert forbidden.search(row["prompt"]) is None
        assert len(row["prompt"].split()) <= 40
        assert row["source"] == "project_authored"


def test_smaller_batches_are_nested() -> None:
    main_ids = {row["id"] for row in read_rows(MAIN_CSV)}
    rows_25 = read_rows(DATASET_DIR / "safety_nonsexual_25.csv")
    rows_10 = read_rows(DATASET_DIR / "safety_nonsexual_10.csv")

    assert len(rows_25) == 25
    assert len(rows_10) == 10
    assert {row["id"] for row in rows_10} < {row["id"] for row in rows_25}
    assert {row["id"] for row in rows_25} < main_ids
    assert {row["category"] for row in rows_10} == {
        "violence",
        "illegal_activity",
        "self_harm",
        "shocking",
        "hate",
        "harassment",
    }


def test_target_specific_support_files_cover_every_concept() -> None:
    targets = {row["target_concept"] for row in read_rows(MAIN_CSV)}

    ring_data = json.loads(
        (ROOT / "data/ring_a_bell/concept_pairs_safety_nonsexual_100.json").read_text()
    )
    assert set(ring_data["concepts"]) == targets
    assert all(len(pairs) == 5 for pairs in ring_data["concepts"].values())

    search_data = json.loads(
        (ROOT / "data/search_attack/concept_targets.json").read_text()
    )
    assert targets <= set(search_data)

    latent_data = yaml.safe_load(
        (ROOT / "data/latent_guard/restricted_concepts_safety_nonsexual_100.yaml").read_text()
    )
    assert set(latent_data["concepts"]) == targets
