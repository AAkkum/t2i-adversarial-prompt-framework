from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any

from t2i_framework.core.types import DefenseDecision


class Defense(ABC):
    """Base interface for prompt and image defense modules."""

    name: str

    def check_prompt(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        return DefenseDecision(allowed=True, reason="not implemented")

    def check_image(
        self,
        image_path: Path,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        return DefenseDecision(allowed=True, reason="not implemented")
