from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


COPRO_SPLITS = ("id", "ood")
COPRO_CONDITIONS = ("explicit", "synonym", "adversarial")

# Latent Guard, ECCV 2024, Table 1b.
PAPER_TABLE_1B_AUC = {
    ("id", "explicit"): 0.985,
    ("id", "synonym"): 0.914,
    ("id", "adversarial"): 0.908,
    ("ood", "explicit"): 0.944,
    ("ood", "synonym"): 0.913,
    ("ood", "adversarial"): 0.915,
}


@dataclass(frozen=True)
class CoProPair:
    unsafe_prompt: str
    safe_prompt: str
    concept: str
    category: str | None = None


def load_copro(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "ID_concepts",
        "OOD_concepts",
        "ID_test_data",
        "OOD_test_data",
        "concept_synonym",
        "concept_adv",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"CoPro dataset is missing required keys: {', '.join(missing)}")
    return data


def concepts_for_split(data: dict[str, Any], split: str) -> list[str]:
    split = _validate_split(split)
    key = "ID_concepts" if split == "id" else "OOD_concepts"
    concepts = data[key]
    if not isinstance(concepts, list) or not all(isinstance(item, str) for item in concepts):
        raise ValueError(f"CoPro field {key} must be a list of strings.")
    return list(concepts)


def pairs_for_condition(
    data: dict[str, Any],
    split: str,
    condition: str,
    limit: int = 0,
) -> list[CoProPair]:
    split = _validate_split(split)
    condition = _validate_condition(condition)
    key = "ID_test_data" if split == "id" else "OOD_test_data"
    raw_entries = data[key]
    if not isinstance(raw_entries, list):
        raise ValueError(f"CoPro field {key} must be a list.")

    synonym_map = data["concept_synonym"]
    adversarial_map = data["concept_adv"]
    replacement_map = synonym_map if condition == "synonym" else adversarial_map
    pairs: list[CoProPair] = []

    for entry in raw_entries:
        pair = _coerce_pair(entry)
        if condition != "explicit":
            if not _eligible_for_replacement(pair, synonym_map, adversarial_map):
                continue
            replacement = _first_replacement(replacement_map[pair.concept])
            pair = CoProPair(
                unsafe_prompt=_replace_concept(pair.unsafe_prompt, pair.concept, replacement),
                safe_prompt=pair.safe_prompt,
                concept=pair.concept,
                category=pair.category,
            )
        pairs.append(pair)
        if limit > 0 and len(pairs) >= limit:
            break
    return pairs


def prediction_rows(
    pairs: Iterable[CoProPair],
    split: str,
    condition: str,
    score_lookup: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs):
        for label, prompt_type, prompt in (
            (1, "unsafe", pair.unsafe_prompt),
            (0, "safe", pair.safe_prompt),
        ):
            rows.append(
                {
                    "split": split,
                    "condition": condition,
                    "pair_index": pair_index,
                    "prompt_type": prompt_type,
                    "label": label,
                    "concept": pair.concept,
                    "category": pair.category,
                    "prompt": prompt,
                    "score": score_lookup[prompt],
                }
            )
    return rows


def binary_metrics(rows: Iterable[dict[str, Any]], threshold: float) -> dict[str, float | int]:
    materialized = list(rows)
    labels = [int(row["label"]) for row in materialized]
    scores = [float(row["score"]) for row in materialized]
    predictions = [int(score >= threshold) for score in scores]
    correct = sum(int(predicted == label) for predicted, label in zip(predictions, labels))
    true_positive = sum(
        int(predicted == 1 and label == 1) for predicted, label in zip(predictions, labels)
    )
    true_negative = sum(
        int(predicted == 0 and label == 0) for predicted, label in zip(predictions, labels)
    )
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    return {
        "examples": len(labels),
        "unsafe_examples": positive_count,
        "safe_examples": negative_count,
        "auc": roc_auc(labels, scores),
        "accuracy": correct / len(labels),
        "unsafe_recall": true_positive / positive_count,
        "safe_recall": true_negative / negative_count,
    }


def roc_auc(labels: list[int], scores: list[float]) -> float:
    if len(labels) != len(scores):
        raise ValueError("ROC-AUC labels and scores must have equal lengths.")
    positive_count = sum(int(label == 1) for label in labels)
    negative_count = sum(int(label == 0) for label in labels)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both positive and negative examples.")

    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][1] == ranked[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for ranked_index in range(start, end):
            original_index = ranked[ranked_index][0]
            ranks[original_index] = average_rank
        start = end

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _coerce_pair(entry: Any) -> CoProPair:
    if not isinstance(entry, dict):
        raise ValueError("Each CoPro test entry must be an object.")
    try:
        unsafe_prompt = entry["unsafe_prompt"]
        safe_prompt = entry["safe_prompt"]
        concept = entry["concept"]
    except KeyError as exc:
        raise ValueError(f"CoPro test entry is missing {exc.args[0]!r}.") from exc
    if not all(isinstance(value, str) and value for value in (unsafe_prompt, safe_prompt, concept)):
        raise ValueError("CoPro prompts and concepts must be non-empty strings.")
    category = entry.get("category")
    return CoProPair(
        unsafe_prompt=unsafe_prompt,
        safe_prompt=safe_prompt,
        concept=concept,
        category=category if isinstance(category, str) else None,
    )


def _eligible_for_replacement(
    pair: CoProPair,
    synonym_map: dict[str, Any],
    adversarial_map: dict[str, Any],
) -> bool:
    return (
        pair.concept in synonym_map
        and pair.concept in adversarial_map
        and (
            pair.concept in pair.unsafe_prompt
            or pair.concept.capitalize() in pair.unsafe_prompt
        )
    )


def _first_replacement(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            raise ValueError("CoPro replacement lists cannot be empty.")
        value = value[0]
    if not isinstance(value, str) or not value:
        raise ValueError("CoPro replacements must be non-empty strings.")
    return value


def _replace_concept(prompt: str, concept: str, replacement: str) -> str:
    if concept in prompt:
        return prompt.replace(concept, replacement)
    capitalized = concept.capitalize()
    if capitalized in prompt:
        return prompt.replace(capitalized, replacement)
    raise ValueError(f"Concept {concept!r} is not present in its unsafe prompt.")


def _validate_split(split: str) -> str:
    normalized = split.lower()
    if normalized not in COPRO_SPLITS:
        raise ValueError(f"Unknown CoPro split {split!r}; expected one of {COPRO_SPLITS}.")
    return normalized


def _validate_condition(condition: str) -> str:
    normalized = condition.lower()
    if normalized not in COPRO_CONDITIONS:
        raise ValueError(
            f"Unknown CoPro condition {condition!r}; expected one of {COPRO_CONDITIONS}."
        )
    return normalized
