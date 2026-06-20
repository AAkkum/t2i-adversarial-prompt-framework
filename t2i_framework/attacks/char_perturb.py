from __future__ import annotations

import random
import unicodedata
from typing import Any

from t2i_framework.attacks.base import Attack
from t2i_framework.core.types import AttackCandidate


class CharPerturbAttack(Attack):
    """Toy character-level perturbations for safe synthetic prompts."""

    name = "char_perturb"

    def generate(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[AttackCandidate]:
        context = context or {}
        rng = random.Random(context.get("seed", 0))
        normalized = unicodedata.normalize("NFKC", prompt)
        leet = normalized.translate(str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0"}))
        tokens = normalized.split()
        if len(tokens) > 1:
            idx = rng.randrange(1, len(tokens))
            spaced = " ".join(tokens[:idx]) + "  " + " ".join(tokens[idx:])
        else:
            spaced = normalized
        zero_width = normalized.replace(" ", "\u200b ", 1) if " " in normalized else normalized
        return [
            AttackCandidate(text=spaced, metadata={"method": "extra_whitespace"}),
            AttackCandidate(text=leet, metadata={"method": "simple_leetspeak"}),
            AttackCandidate(text=zero_width, metadata={"method": "unicode_spacing_example"}),
        ]
