from __future__ import annotations

from typing import Any

from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense


class EmbeddingFilterDefense(Defense):
    """Placeholder for future transformer-based text embedding similarity filtering."""

    name = "embedding_filter"

    def check_prompt(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        return DefenseDecision(
            allowed=True,
            reason="embedding filter placeholder; install optional model dependencies later",
            score=None,
        )
