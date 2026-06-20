from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml

from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense


def normalize_text(text: str, remove_spaces: bool = False) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if remove_spaces:
        normalized = normalized.replace(" ", "")
    return normalized


class NormalizeKeywordsDefense(Defense):
    """Simple filter for safe synthetic restricted concepts and aliases."""

    name = "normalize_keywords"

    def __init__(self, concepts_path: Path | None = None, remove_spaces: bool = True) -> None:
        self.concepts_path = concepts_path or Path("data/restricted_concepts.yaml")
        self.remove_spaces = remove_spaces
        self.concepts = self._load_concepts(self.concepts_path)

    @staticmethod
    def _load_concepts(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data.get("concepts", {})

    def _restricted_terms(self, target_concept: str | None) -> list[str]:
        terms: list[str] = []
        if target_concept:
            terms.append(target_concept)
        for concept, config in self.concepts.items():
            terms.append(concept)
            terms.extend(config.get("aliases", []))
        return sorted(set(terms))

    def check_prompt(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        normalized_prompt = normalize_text(prompt)
        compact_prompt = normalize_text(prompt, remove_spaces=self.remove_spaces)
        for term in self._restricted_terms(target_concept):
            normalized_term = normalize_text(term)
            compact_term = normalize_text(term, remove_spaces=self.remove_spaces)
            if normalized_term in normalized_prompt or compact_term in compact_prompt:
                return DefenseDecision(
                    allowed=False,
                    reason=f"matched restricted concept or alias: {term}",
                    score=1.0,
                    metadata={"matched_term": term},
                )
        return DefenseDecision(allowed=True, reason="no restricted concept matched", score=0.0)
