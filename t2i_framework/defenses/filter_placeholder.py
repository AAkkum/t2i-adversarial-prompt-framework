from __future__ import annotations

from pathlib import Path
from typing import Any

from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense


class FilterPlaceholderDefense(Defense):
    """Neutral oracle placeholder; it deliberately implements no safety logic."""

    name = "filter_placeholder"
    _reason = "Placeholder filter: no blocking rule configured"

    def check_prompt(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        return DefenseDecision(allowed=True, reason=self._reason, score=1.0)

    def check_image(
        self,
        image_path: Path,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        return DefenseDecision(allowed=True, reason=self._reason, score=1.0)
