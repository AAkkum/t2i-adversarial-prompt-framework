from __future__ import annotations

from typing import Any

from t2i_framework.evaluation.prompt_similarity import PromptSimilarityEvaluator


class EvaluationSuite:
    """Collect optional evaluation metrics for a completed run."""

    def __init__(self, prompt_similarity: PromptSimilarityEvaluator | None = None) -> None:
        self.prompt_similarity = prompt_similarity or PromptSimilarityEvaluator()

    def evaluate(
        self,
        original_prompt: str,
        attacked_prompt: str,
        image_path: str | None,
        config: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        evaluation_config = dict(config.get("evaluation", {}))
        scores: dict[str, float] = {}
        metadata: dict[str, Any] = {}

        prompt_scores, prompt_metadata = self.prompt_similarity.evaluate(
            original_prompt,
            attacked_prompt,
            evaluation_config,
        )
        scores.update(prompt_scores)
        metadata.update(prompt_metadata)

        return scores, metadata
