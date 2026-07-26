from __future__ import annotations

import re
import unicodedata
from typing import Any

from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense


def normalize_character_text(text: str) -> str:
    """Normalize case, Unicode representation, and whitespace for matching."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


class CharacterFilterDefense(Defense):
    """Controlled local keyword filter for the Mario experiment."""

    name = "character_filter"
    blocked_terms = ("mushroom kingdom", "super mario", "nintendo", "mario")

    def check_prompt(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        normalized_prompt = normalize_character_text(prompt)
        for term in self.blocked_terms:
            pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
            if re.search(pattern, normalized_prompt):
                return DefenseDecision(
                    allowed=False,
                    reason=f"Blocked character term: {term}",
                    score=1.0,
                    metadata={"matched_term": term},
                )
        return DefenseDecision(
            allowed=True,
            reason="No blocked character term found",
            score=0.0,
        )
