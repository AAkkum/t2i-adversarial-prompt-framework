"""Offline data coverage, provenance and lookup contracts; no pretrained models."""

import csv
import hashlib
import io
import json
from collections import Counter
from copy import deepcopy

import pytest

from scripts.build_ring_a_bell_concept_pairs import (
    AUDIT_PATH,
    CSV_PATH,
    PAIR_PATH,
    SPEC_PATH,
    build,
    render,
)
from t2i_framework.attacks.ring_a_bell_encoder import load_pairs
from t2i_framework.attacks.search_attack import SearchAttack

SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
TARGETS = ["red", *SPEC["subjects"]]


@pytest.mark.parametrize("target", TARGETS)
def test_all_configured_concepts_have_valid_controlled_pairs(target):
    pairs, digest = load_pairs(PAIR_PATH, target)
    assert len(pairs) == 12
    assert digest == hashlib.sha256(PAIR_PATH.read_bytes()).hexdigest()
    assert all(p.strip() and n.strip() and p != n for p, n in pairs)
    assert len(set(pairs)) == 12
    if target != "red":
        subject = SPEC["subjects"][target]
        for index, (positive, negative) in enumerate(pairs):
            template = SPEC["templates"][index]
            assert positive == template.format(subject=subject["positive"])
            assert negative == template.format(subject=subject["negatives"][index % 3])


@pytest.mark.parametrize(
    "variant, canonical",
    [
        ("RED", "red"),
        ("Dog", "dog"),
        ("Donald Trump", "donald trump"),
        ("DONALD TRUMP", "donald trump"),
        ("  Donald\t Trump  ", "donald trump"),
        ("ＢＡＲＡＣＫ ＯＢＡＭＡ", "barack obama"),
        ("Barack Obama", "barack obama"),
        ("SINGAPORE_AIRLINES", "singapore_airlines"),
    ],
)
def test_deterministic_case_unicode_whitespace_lookup(variant, canonical):
    assert load_pairs(PAIR_PATH, variant) == load_pairs(PAIR_PATH, canonical)


@pytest.mark.parametrize(
    "unknown",
    [
        "dogg",
        "trump",
        "DonaldTrump",
        "Donald Trump and Barack Obama",
        "singapore airlines",
        "not configured",
    ],
)
def test_no_fuzzy_matching_or_combined_identity(unknown):
    with pytest.raises(ValueError, match="No positive/negative pairs configured"):
        load_pairs(PAIR_PATH, unknown)


def test_normalized_collisions_fail_instead_of_silently_selecting(tmp_path):
    path = tmp_path / "pairs.json"
    entries = [{"positive": "one", "negative": "two"}]
    path.write_text(json.dumps({"concepts": {"dog": entries, " DOG ": entries}}))
    with pytest.raises(ValueError, match="Ambiguous concept keys"):
        load_pairs(path, "dog")


def test_preserved_red_and_separate_identities():
    data = json.loads(PAIR_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(
        json.dumps(data["concepts"]["red"], sort_keys=True).encode()
    ).hexdigest() == ("e9a485318c70cd416da0aa1d4475eb820a2e176bfa609a21c53bd8d51de8c19a")
    trump, _ = load_pairs(PAIR_PATH, "donald trump")
    obama, _ = load_pairs(PAIR_PATH, "barack obama")
    assert all("Donald Trump" in p and "Obama" not in p + n for p, n in trump)
    assert all("Barack Obama" in p and "Trump" not in p + n for p, n in obama)
    assert "kissing" not in PAIR_PATH.read_text(encoding="utf-8")
    assert data["provenance"]["kind"] == "PROJECT-CREATED CONCEPT PAIRS"
    assert data["provenance"]["author_data_included"] is False


def test_reproducible_outputs_and_complete_reviewed_source_coverage():
    pairs, audit_bytes = render()
    assert pairs == PAIR_PATH.read_bytes()
    assert audit_bytes == AUDIT_PATH.read_bytes()
    source = list(csv.DictReader(io.StringIO(CSV_PATH.read_text(encoding="utf-8-sig"))))
    audit = list(csv.DictReader(io.StringIO(audit_bytes.decode("utf-8"))))
    assert len(source) == len(audit) == 100
    assert {r["target_concept"] for r in source} == set(TARGETS) - {"red"}
    assert [r["id"] for r in source] == [r["id"] for r in audit]
    assert [r["prompt"] for r in source] == [r["prompt"] for r in audit]
    assert Counter(r["target_presence"] for r in audit) == {
        "explicit_target": 93,
        "implicit_ambiguous": 6,
        "conflicting_cue": 1,
    }
    assert {r["target_concept"] for r in audit if r["target_presence"] != "explicit_target"} == {
        "adidas",
        "apple",
        "audi",
        "bmw",
        "coca-cola",
        "puma",
        "singapore_airlines",
    }
    assert all(r["concept_supported"] == "True" and r["pair_count"] == "12" for r in audit)
    texts = {
        text for target in TARGETS for pair in load_pairs(PAIR_PATH, target)[0] for text in pair
    }
    assert not texts.intersection(r["prompt"] for r in source)


def test_source_drift_and_missing_subjects_require_review():
    with pytest.raises(ValueError, match="CSV differs"):
        build(SPEC, CSV_PATH.read_bytes() + b"\n")
    incomplete = deepcopy(SPEC)
    del incomplete["subjects"]["dog"]
    with pytest.raises(ValueError, match="cover exactly"):
        build(incomplete, CSV_PATH.read_bytes())


def test_search_attack_retains_original_target_contract():
    prompt = "a person holding an umbrella while walking through a futuristic city at night"
    context = {"seed": 42, "max_candidates": 3}
    before = deepcopy(context)
    candidates = SearchAttack().generate(prompt, "umbrella", context)
    assert candidates[0].text == prompt
    assert len(candidates) == 3
    assert all("handheld rain protection canopy" in c.text for c in candidates[1:])
    assert context == before


@pytest.mark.parametrize("defect", ["duplicate_template", "duplicate_negative", "target_leakage"])
def test_generator_rejects_pair_quality_regressions(defect):
    spec = deepcopy(SPEC)
    if defect == "duplicate_template":
        spec["templates"][1] = spec["templates"][0]
        message = "distinct templates"
    elif defect == "duplicate_negative":
        spec["subjects"]["dog"]["negatives"][1] = " A CAT "
        message = "Duplicate negative"
    else:
        spec["subjects"]["dog"]["negatives"][0] = "a dog beside a cat"
        message = "explicitly contains"
    with pytest.raises(ValueError, match=message):
        build(spec, CSV_PATH.read_bytes())


def test_reviewed_controls_do_not_add_demographics_or_identifying_emblems():
    subjects = SPEC["subjects"]
    for target in list(subjects)[68:]:
        positive = subjects[target]["positive"]
        assert positive.startswith("a person identified as ")
        for negative in subjects[target]["negatives"]:
            assert negative.startswith("a person ")
            assert set(negative.split()) <= {
                "a",
                "person",
                "with",
                "an",
                "unspecified",
                "identity",
                "whose",
                "name",
                "is",
                "without",
                "specified",
            }
    for target in list(subjects)[40:52]:
        definition = subjects[target]
        product = definition["positive"].split(" with ")[0]
        assert all(n.startswith(product + " with no brand") for n in definition["negatives"])
    assert subjects["bmw"]["positive"].startswith("an SUV ")
    for target in list(subjects)[52:68]:
        definition = subjects[target]
        kind = definition["positive"].split(" identified as ")[0]
        assert all(n.startswith(kind + " ") for n in definition["negatives"])
        assert not any(
            cue in n
            for n in definition["negatives"]
            for cue in ("echidna", "toy astronaut", "armored", "Sonic", "Nintendo", "Disney")
        )
