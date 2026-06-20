from __future__ import annotations

from typing import Any

from t2i_framework.attacks.base import Attack
from t2i_framework.core.types import AttackCandidate


class SearchAttack(Attack):
    """Placeholder for future query-based candidate search."""

    name = "search_attack"

    def generate(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[AttackCandidate]:
        return [
            AttackCandidate(
                text=prompt,
                metadata={"status": "stub", "planned_method": "query-based candidate search"},
            )
        ]
