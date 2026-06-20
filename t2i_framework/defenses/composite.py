from __future__ import annotations

from pathlib import Path
from typing import Any

from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense
from t2i_framework.defenses.embedding_filter import EmbeddingFilterDefense
from t2i_framework.defenses.image_clip_filter import ImageClipFilterDefense
from t2i_framework.defenses.normalize_keywords import NormalizeKeywordsDefense


class CompositeDefense(Defense):
    """Run multiple defenses and block when any sub-defense blocks."""

    name = "composite"

    def __init__(self) -> None:
        self.prompt_defenses = [NormalizeKeywordsDefense(), EmbeddingFilterDefense()]
        self.image_defenses = [ImageClipFilterDefense()]

    def check_prompt(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        decisions = [
            defense.check_prompt(prompt, target_concept=target_concept, context=context)
            for defense in self.prompt_defenses
        ]
        blocked = next((decision for decision in decisions if not decision.allowed), None)
        if blocked:
            blocked.metadata["sub_decisions"] = [decision.reason for decision in decisions]
            return blocked
        return DefenseDecision(
            allowed=True,
            reason="all prompt sub-defenses allowed",
            metadata={"sub_decisions": [decision.reason for decision in decisions]},
        )

    def check_image(
        self,
        image_path: Path,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        decisions = [
            defense.check_image(image_path, target_concept=target_concept, context=context)
            for defense in self.image_defenses
        ]
        blocked = next((decision for decision in decisions if not decision.allowed), None)
        if blocked:
            blocked.metadata["sub_decisions"] = [decision.reason for decision in decisions]
            return blocked
        return DefenseDecision(
            allowed=True,
            reason="all image sub-defenses allowed",
            metadata={"sub_decisions": [decision.reason for decision in decisions]},
        )
