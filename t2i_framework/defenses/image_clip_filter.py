from __future__ import annotations

from pathlib import Path
from typing import Any

from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense
from t2i_framework.evaluation.clip_image_text import CLIPImageTextScorer, ImageTextScorer


class ImageClipFilterDefense(Defense):
    """CLIP-based image-level target concept filter."""

    name = "image_clip_filter"

    def __init__(
        self,
        threshold: float = 0.25,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
        scorer: ImageTextScorer | None = None,
    ) -> None:
        self.threshold = threshold
        self.model_id = model_id
        self.device = device
        self._scorer = scorer

    def check_image(
        self,
        image_path: Path,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        if not target_concept:
            return DefenseDecision(
                allowed=True,
                reason="image CLIP filter skipped because no target concept was provided",
                score=None,
            )

        config = (context or {}).get("config", {})
        clip_config = dict(config.get("evaluation", {}).get("image_clip", {}))
        clip_config.update(dict(config.get("defense", {}).get("image_clip", {})))
        threshold = float(clip_config.get("threshold", self.threshold))
        model_id = clip_config.get("model_id", self.model_id)
        scorer = (
            (context or {}).get("image_text_scorer")
            or clip_config.get("scorer")
            or self._get_scorer(model_id, clip_config.get("device", self.device))
        )
        similarity = scorer.score(image_path, target_concept)
        if similarity >= threshold:
            return DefenseDecision(
                allowed=False,
                reason="image matched target concept above CLIP threshold",
                score=similarity,
                metadata={
                    "target_concept": target_concept,
                    "threshold": threshold,
                    "model_id": model_id,
                },
            )

        return DefenseDecision(
            allowed=True,
            reason="image target similarity below CLIP threshold",
            score=similarity,
            metadata={
                "target_concept": target_concept,
                "threshold": threshold,
                "model_id": model_id,
            },
        )

    def _get_scorer(self, model_id: str, device: str | None) -> ImageTextScorer:
        if self._scorer is None:
            self._scorer = CLIPImageTextScorer(model_id=model_id, device=device)
        return self._scorer
