from __future__ import annotations

from pathlib import Path
from typing import Any

from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense


class NoneDefense(Defense):
    """Defense that always allows prompts and images."""

    name = "none"

    def check_prompt(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        return DefenseDecision(allowed=True, reason="no defense configured")

    def check_image(
        self,
        image_path: Path,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        return DefenseDecision(allowed=True, reason="no defense configured")
