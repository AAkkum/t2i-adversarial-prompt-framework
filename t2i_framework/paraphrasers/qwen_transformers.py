from __future__ import annotations

from t2i_framework.llm_backends.qwen_transformers import QwenTransformersBackend
from t2i_framework.paraphrasers.qwen_ollama import (
    build_paraphrase_prompt,
    normalize_detail_level,
    parse_candidate_list,
)


class QwenTransformersParaphraser:
    """Hugging Face Transformers provider for Qwen paraphrase candidates."""

    def __init__(
        self,
        backend: QwenTransformersBackend,
        detail_level: str = "compact",
        max_new_tokens: int = 512,
    ) -> None:
        self.backend = backend
        self.model = backend.model_id
        self.detail_level = normalize_detail_level(detail_level)
        self.max_new_tokens = max_new_tokens
        self.last_raw_response: str | None = None
        self.last_candidates: list[str] = []

    def generate_candidates(
        self,
        concept: str,
        context: str = "",
        count: int = 5,
    ) -> list[str]:
        prompt = build_paraphrase_prompt(concept, context, count, self.detail_level)
        response = self.backend.generate(prompt, max_new_tokens=self.max_new_tokens)
        candidates = parse_candidate_list(response, limit=count)
        self.last_raw_response = response
        self.last_candidates = candidates
        return candidates

    def unload(self) -> None:
        self.backend.unload()
