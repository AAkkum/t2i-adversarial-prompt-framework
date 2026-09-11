from __future__ import annotations

from typing import Any

from t2i_framework.core.logging_utils import console
from t2i_framework.evaluation.metrics import score_band
from t2i_framework.similarity.sentence_text import SentenceTextSimilarityScorer


DEFAULT_PROMPT_SIMILARITY_BANDS = {
    "strong": 0.8,
    "moderate": 0.65,
    "weak": 0.5,
}


class PromptSimilarityEvaluator:
    """Evaluate semantic similarity between original and attacked prompts."""

    def __init__(self, scorer: Any | None = None) -> None:
        self._scorer = scorer
        self._warning_printed = False

    def evaluate(
        self,
        original_prompt: str,
        attacked_prompt: str,
        config: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        prompt_config = dict(config.get("prompt_similarity", {}))
        enabled = bool(prompt_config.get("enabled", False))
        if not enabled:
            return {}, {"prompt_similarity": {"enabled": False}}

        model_id = str(
            prompt_config.get("model_id", "sentence-transformers/all-MiniLM-L6-v2")
        )
        device = str(prompt_config.get("device", "cpu"))
        bands = dict(prompt_config.get("bands", DEFAULT_PROMPT_SIMILARITY_BANDS))

        try:
            if self._scorer is None:
                self._scorer = SentenceTextSimilarityScorer(
                    model_id=model_id,
                    device=device,
                )
            score = self._scorer.score(original_prompt, attacked_prompt)
        except RuntimeError as exc:
            if not self._warning_printed:
                console.print(f"[evaluation] prompt similarity disabled: {exc}")
                self._warning_printed = True
            return {}, {
                "prompt_similarity": {
                    "enabled": False,
                    "reason": str(exc),
                }
            }

        return {
            "prompt_prompt_similarity": score,
        }, {
            "prompt_similarity": {
                "enabled": True,
                "method": "sentence_transformer",
                "model_id": model_id,
                "device": device,
                "score": score,
                "quality_band": score_band(score, bands),
                "bands": bands,
            }
        }
