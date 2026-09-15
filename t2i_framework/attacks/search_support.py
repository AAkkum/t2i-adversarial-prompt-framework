"""Data loading and phrase construction helpers for search-based components."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data/search_attack"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def load_concept_targets(path: Path | None = None) -> dict[str, str]:
    source = path if path is not None else DATA_DIR / "concept_targets.json"
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or any(
        not isinstance(key, str) or not key.strip()
        or not isinstance(value, str) or not value.strip()
        for key, value in data.items()
    ):
        raise ValueError("concept_targets.json must map non-empty terms to descriptions.")
    return data


def replace_concepts(text: str, mapping: dict[str, str]) -> str:
    """Replace whole phrases once and keep a preceding English article valid."""
    if not mapping:
        return text
    ordered = sorted(mapping, key=lambda key: (-len(key), key.casefold()))
    lookup = {key.casefold(): value for key, value in mapping.items()}
    pattern = re.compile(
        r"(?:(?P<article>\b(?:a|an))\s+)?"
        r"(?P<term>(?<!\w)(?:" + "|".join(map(re.escape, ordered)) + r")(?!\w))",
        re.IGNORECASE,
    )

    def replacement(match: re.Match[str]) -> str:
        description = lookup[match.group("term").casefold()]
        article = match.group("article")
        if article is None:
            return description
        corrected = "an" if description[0].casefold() in "aeiou" else "a"
        if article[0].isupper():
            corrected = corrected.capitalize()
        return f"{corrected} {description}"

    return pattern.sub(replacement, text)


def scene_description(prompt: str) -> str:
    # A small grammatical heuristic, not a complete English parser.
    match = re.search(
        r"\b(?:standing|walking|sitting|running|displayed|holding|posing|parked|working|"
        r"resting|lying|flying|floating|wearing|riding)\b.*", prompt, re.IGNORECASE
    )
    if match is None:
        match = re.search(
            r"\b(?:in|on|at|beside|near|under|above|inside|outside)\b.*", prompt, re.IGNORECASE
        )
    return match.group().strip().rstrip(".!? ") if match else ""


def semantic_base(
    prompt: str,
    target: str | None,
    mapping: dict[str, str],
    replacement_concept: str | None = None,
    *,
    target_mode: str = "blocked_term",
) -> str:
    """Build the base subject/scene text for search_attack variants."""
    if target is None:
        return replace_concepts(prompt.strip(), mapping)

    if target_mode == "replacement":
        return _merge_subject_and_scene(prompt, target, mapping)
    if target_mode != "blocked_term":
        raise ValueError("target_mode must be either 'blocked_term' or 'replacement'.")

    replacement = _replacement_for_target(target, mapping, replacement_concept)
    if replacement is None:
        return replace_concepts(prompt.strip(), mapping).rstrip(".!? ")

    target_mapping = _target_alias_mapping(target, replacement, mapping)
    replaced_prompt = replace_concepts(prompt.strip(), target_mapping)
    if replaced_prompt != prompt.strip():
        return replaced_prompt.rstrip(".!? ")

    return _merge_subject_and_scene(prompt, replacement, mapping)


def _replacement_for_target(
    target: str,
    mapping: dict[str, str],
    replacement_concept: str | None,
) -> str | None:
    if isinstance(replacement_concept, str) and replacement_concept.strip():
        return replacement_concept.strip().rstrip(".!? ")

    normalized_target = normalize_text(target)
    for term, description in mapping.items():
        if normalize_text(term) == normalized_target:
            return description
    return None


def _target_alias_mapping(
    target: str,
    replacement: str,
    mapping: dict[str, str],
) -> dict[str, str]:
    normalized_target = normalize_text(target)
    normalized_replacement = normalize_text(replacement)
    aliases = {
        term: description
        for term, description in mapping.items()
        if normalize_text(term) == normalized_target
        or normalize_text(description) == normalized_replacement
    }
    aliases.setdefault(target, replacement)
    return aliases


def _merge_subject_and_scene(
    prompt: str,
    subject: str,
    mapping: dict[str, str],
) -> str:
    subject = subject.strip().rstrip(".!? ")
    normalized_prompt = normalize_text(prompt)
    normalized_target = normalize_text(subject)
    if f" {normalized_target} " in f" {normalized_prompt} ":
        return prompt.strip().rstrip(".!? ")

    matching_concepts = {
        term: description
        for term, description in mapping.items()
        if normalized_target == normalize_text(description)
        or normalized_target.startswith(f"{normalize_text(description)} ")
    }
    replaced_prompt = replace_concepts(prompt.strip(), matching_concepts)
    if replaced_prompt != prompt.strip():
        return replaced_prompt.rstrip(".!? ")

    scene = replace_concepts(scene_description(prompt), mapping)
    subject_tokens = normalize_text(subject).split()
    scene_words = scene.split()
    scene_tokens = [normalize_text(word) for word in scene_words]
    if scene_tokens and " ".join(scene_tokens) in " ".join(subject_tokens):
        return subject
    overlap = 0
    for length in range(1, min(len(subject_tokens), len(scene_tokens)) + 1):
        if subject_tokens[-length:] == scene_tokens[:length]:
            overlap = length
    # Do not append a second action when the explicit target already contains one.
    if overlap == 0 and scene_description(subject):
        return subject
    return " ".join([subject, *scene_words[overlap:]]).strip()
