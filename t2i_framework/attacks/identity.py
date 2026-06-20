from __future__ import annotations

from typing import Any

from t2i_framework.attacks.base import Attack
from t2i_framework.core.types import AttackCandidate


class IdentityAttack(Attack):
    """Return the input prompt unchanged."""

    name = "identity"

    def generate(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[AttackCandidate]:
        return [AttackCandidate(text=prompt, metadata={"method": "identity"})]
