from __future__ import annotations

from t2i_framework.judges.ollama_similarity import build_similarity_judge_prompt, parse_score
from t2i_framework.llm_backends.qwen_transformers import QwenTransformersBackend


class TransformersSimilarityJudge:
    """In-process Transformers semantic-preservation judge."""

    def __init__(
        self,
        backend: QwenTransformersBackend,
        max_new_tokens: int = 128,
    ) -> None:
        self.backend = backend
        self.model = backend.model_id
        self.max_new_tokens = max_new_tokens

    def score(self, source: str, candidate: str, context: str = "") -> float:
        prompt = build_similarity_judge_prompt(source, candidate, context)
        return parse_score(
            self.backend.generate(
                prompt,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
            )
        )

    def unload(self) -> None:
        self.backend.unload()
