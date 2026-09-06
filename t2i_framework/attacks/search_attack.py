from __future__ import annotations

import json
import random
import re
import unicodedata
from typing import Any

from t2i_framework.attacks.base import Attack
from t2i_framework.core.types import AttackCandidate
from t2i_framework.search_support import (
    DATA_DIR,
    load_concept_targets,
    semantic_base,
)

VARIANT_POOL_SIZE = 400

# Load editable prompt phrases.
with (DATA_DIR / "variant_phrases.json").open(encoding="utf-8") as handle:
    CATEGORY_PHRASES = {key: tuple(values) for key, values in json.load(handle).items()}

# Do not combine categories that describe the same visual choice.
_EXCLUSIVE_GROUPS = (
    {"perspective", "camera_angle"},
    {"camera_distance", "composition"},
    {"background", "depth_of_field"},
    {"lighting", "time_of_day"},
)
MAX_VARIANT_WORDS = 80


class SearchAttack(Attack):
    """This is a theoretical search-attack prototype with data-driven concepts."""

    name = "search_attack"

    def generate(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[AttackCandidate]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("SearchAttack requires a non-empty prompt.")
        if target_concept is not None and (
            not isinstance(target_concept, str) or not target_concept.strip()
        ):
            raise ValueError("target_concept must be non-empty text or None.")

        context = context or {}
        max_candidates = context.get("max_candidates", 5)
        if type(max_candidates) is not int or not 1 <= max_candidates <= 20:
            raise ValueError("max_candidates must be an integer between 1 and 20.")

        candidates = [
            AttackCandidate(
                text=prompt,
                metadata={"method": "original", "candidate_index": 0},
            )
        ]
        if max_candidates == 1:
            return candidates
        mapping = load_concept_targets()
        base = semantic_base(prompt, target_concept, mapping)
        variant_pool = _build_variant_pool(base, "", seed=context.get("seed", 0))
        for text, categories in variant_pool[: max_candidates - 1]:
            candidates.append(
                AttackCandidate(
                    text=text,
                    metadata={
                        "method": "semantic_combination",
                        "categories": categories,
                        "candidate_index": len(candidates),
                    },
                )
            )
        return candidates


def _build_variant_pool(
    subject: str,
    scene: str,
    seed: int,
    pool_size: int = VARIANT_POOL_SIZE,
) -> list[tuple[str, list[str]]]:
    """Build a deterministic pool whose prompts combine 2–4 category modifiers."""

    if pool_size < 100:
        raise ValueError("The internal variant pool must contain at least 100 variants.")
    rng = random.Random(seed)
    base = " ".join(f"{subject} {scene}".split()).rstrip(".!? ")
    category_names = list(CATEGORY_PHRASES)
    # These categories require matching words in the source prompt.
    prerequisites = {
        "clothing": r"\b(?:wearing|clothing|clothes|shirt|overalls|jacket|dressed)\b",
        "facial_expression": r"\b(?:face|facial|eyes|expression|smile|mustached)\b",
        "weather": r"\b(?:rain|snow|sunshine|weather|clouds|fog)\b",
        "time_of_day": r"\b(?:morning|afternoon|evening|night|dawn|sunset|daylight)\b",
    }
    category_names = [
        name for name in category_names
        if name not in prerequisites or re.search(prerequisites[name], base, re.IGNORECASE)
    ]
    if len(base.split()) > MAX_VARIANT_WORDS - 20:
        raise ValueError("Prompt/target is too long for concise variants; shorten it to 60 words.")
    variants: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    signatures: list[set[str]] = []
    attempts = 0
    max_attempts = pool_size * 100

    while len(variants) < pool_size and attempts < max_attempts:
        attempts += 1
        selected_categories = rng.sample(category_names, rng.randint(2, 4))
        if any(len(group.intersection(selected_categories)) > 1 for group in _EXCLUSIVE_GROUPS):
            continue
        modifiers = [rng.choice(CATEGORY_PHRASES[name]) for name in selected_categories]
        # Compare modifiers without the shared base to detect near-duplicates.
        signature = set(_normalized_candidate(" ".join(modifiers)).split())
        if any(len(signature & previous) / len(signature | previous) >= 0.85
               for previous in signatures):
            continue
        template = rng.randrange(3)
        if template == 0:
            text = f"{base}, {', '.join(modifiers)}."
        elif template == 1:
            text = f"{base}. Visual treatment: {', '.join(modifiers)}."
        else:
            text = f"{base}, {', '.join(modifiers[:-1])} and {modifiers[-1]}."
        if len(text.split()) > MAX_VARIANT_WORDS:
            continue
        normalized = _normalized_candidate(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        signatures.append(signature)
        variants.append((text, selected_categories))

    if len(variants) != pool_size:
        raise RuntimeError(f"Could only generate {len(variants)} unique internal variants.")
    return variants


def _normalized_candidate(text: str) -> str:
    """Normalize case, whitespace, and non-semantic punctuation for deduplication."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()
