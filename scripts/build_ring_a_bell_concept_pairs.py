"""Rebuild PROJECT-CREATED CONCEPT PAIRS and their pinned representative CSV audit.

This offline data-preparation utility is not part of Ring-A-Bell's search.
It uses reviewed subject contrasts, not the CSV's evaluation prompts, to build pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "data/ring_a_bell/concept_pair_spec.json"
CSV_PATH = ROOT / "data/datasets/representative/representative_prompt_batch_100.csv"
PAIR_PATH = ROOT / "data/ring_a_bell/concept_pairs.json"
AUDIT_PATH = ROOT / "data/ring_a_bell/representative_target_audit.csv"


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def contains_target(prompt: str, target: str) -> bool:
    # Underscores are separators for this audit only, not a runtime lookup alias.
    term = normalize(target).replace("_", " ")
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalize(prompt)) is not None


def build(spec: dict[str, Any], csv_bytes: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_hash = hashlib.sha256(csv_bytes).hexdigest()
    if source_hash != spec["source_csv_sha256"]:
        raise ValueError(
            "CSV differs from the reviewed source; review it and update the spec first."
        )
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
    targets = [row["target_concept"] for row in rows]
    if len(rows) != 100 or len(set(targets)) != 100:
        raise ValueError("Expected the reviewed 100 cases with 100 distinct targets.")
    if set(spec["subjects"]) != set(targets):
        raise ValueError(
            "Subject specifications must cover exactly the representative CSV targets."
        )
    if set(spec["preserved_pairs"]) & set(targets):
        raise ValueError("Preserved pairs must not overwrite representative concept definitions.")

    concepts = dict(spec["preserved_pairs"])
    templates = spec["templates"]
    if (
        len(templates) != 12
        or len(set(templates)) != 12
        or any(template.count("{subject}") != 1 for template in templates)
    ):
        raise ValueError("Expected 12 distinct templates with exactly one subject slot each.")
    for target in targets:
        subjects = spec["subjects"][target]
        positive = subjects["positive"]
        negatives = subjects["negatives"]
        if (
            not isinstance(positive, str)
            or not positive.strip()
            or not isinstance(negatives, list)
            or len(negatives) != 3
        ):
            raise ValueError(f"Invalid subject definition for {target}.")
        if any(not isinstance(n, str) or not n.strip() or n == positive for n in negatives):
            raise ValueError(f"Invalid negative subjects for {target}.")
        if len({normalize(n) for n in negatives}) != 3:
            raise ValueError(f"Duplicate negative subjects for {target}.")
        if any(contains_target(n, target) for n in negatives):
            raise ValueError(f"Negative subject explicitly contains the target {target!r}.")
        concepts[target] = [
            {
                "positive": template.format(subject=positive),
                "negative": template.format(subject=negatives[index % len(negatives)]),
            }
            for index, template in enumerate(templates)
        ]

    audit = []
    reviewed = set()
    for row in rows:
        target = row["target_concept"]
        review = spec["semantic_reviews"].get(row["id"])
        if review:
            status, evidence, note = review["status"], review["evidence"], review["note"]
            reviewed.add(row["id"])
        elif contains_target(row["prompt"], target):
            status = "explicit_target"
            evidence = target
            note = (
                "Target is explicit after case/Unicode/whitespace normalization; context reviewed."
            )
        else:
            raise ValueError(f"Manual semantic review required for {row['id']}.")
        audit.append(
            {
                "id": row["id"],
                "target_concept": target,
                "category": row["category"],
                "dataset": row["dataset"],
                "prompt": row["prompt"],
                "target_presence": status,
                "evidence": evidence,
                "review_note": note,
                "concept_supported": True,
                "pair_count": len(concepts[target]),
            }
        )
    if reviewed != set(spec["semantic_reviews"]):
        raise ValueError("Semantic review contains cases absent from the source CSV.")

    data = {
        "description": "PROJECT-CREATED CONCEPT PAIRS. No author dataset is included.",
        "provenance": {
            "kind": "PROJECT-CREATED CONCEPT PAIRS",
            "recipe": "data/ring_a_bell/concept_pair_spec.json",
            "recipe_sha256": hashlib.sha256(
                json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "recipe_hash_encoding": "json.dumps(spec, sort_keys=True, ensure_ascii=False), UTF-8",
            "generator": "scripts/build_ring_a_bell_concept_pairs.py",
            "source_csv": spec["source_csv"],
            "provided_filename": spec["provided_filename"],
            "source_csv_sha256": source_hash,
            "method": "Reviewed subject contrasts in identical contexts; 3 negative wordings cycled over 12 contexts.",
            "quality_review": spec["quality_review"],
            "preserved_concepts": list(spec["preserved_pairs"]),
            "author_data_included": False,
            "image_validated": False,
            "evaluation_prompt_text_used_for_pairs": False,
        },
        "concepts": concepts,
    }
    return data, audit


def render(spec_path: Path = SPEC_PATH, csv_path: Path = CSV_PATH) -> tuple[bytes, bytes]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    data, audit = build(spec, csv_path.read_bytes())
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(audit[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(audit)
    return (
        (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        output.getvalue().encode("utf-8"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--out", type=Path, default=PAIR_PATH)
    parser.add_argument("--audit-out", type=Path, default=AUDIT_PATH)
    parser.add_argument("--check", action="store_true", help="Verify outputs without writing them.")
    args = parser.parse_args()
    pairs, audit = render(args.spec, args.csv)
    for path, payload in ((args.out, pairs), (args.audit_out, audit)):
        if args.check:
            if not path.exists() or path.read_bytes() != payload:
                raise SystemExit(f"Generated file differs: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    records = list(csv.DictReader(io.StringIO(audit.decode("utf-8"))))
    print(
        f"100/100 representative concepts supported; {dict(Counter(r['target_presence'] for r in records))}"
    )
    print(
        "Verified generated files."
        if args.check
        else "Wrote project pairs and audit; no CSV cases changed."
    )


if __name__ == "__main__":
    main()
