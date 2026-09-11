"""CPU-only MiniLM matching for the local multi-stage defense."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class SemanticMatch:
    blocked_term: str
    protected_concept: str
    similarity: float


class MiniLMConceptMatcher:
    """Lazily match text against data-driven protected concept descriptions."""

    def __init__(self) -> None:
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(MINILM_MODEL_ID)
            self._model = AutoModel.from_pretrained(MINILM_MODEL_ID).to("cpu")
            self._model.eval()

    def match(self, text: str, concepts: dict[str, str]) -> SemanticMatch:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("MiniLM requires non-empty text.")
        if not concepts:
            raise ValueError("MiniLM requires at least one protected concept.")

        terms = list(concepts)
        descriptions = [concepts[term] for term in terms]
        similarities = self._similarities(text, descriptions)
        if len(similarities) != len(descriptions) or not all(
            isfinite(score) for score in similarities
        ):
            raise RuntimeError("MiniLM returned invalid similarity values.")
        best_index = max(range(len(similarities)), key=similarities.__getitem__)
        return SemanticMatch(
            blocked_term=terms[best_index],
            protected_concept=descriptions[best_index],
            similarity=similarities[best_index],
        )

    def _similarities(self, text: str, descriptions: list[str]) -> list[float]:
        import torch
        from torch.nn import functional

        self._load()
        encoded = self._tokenizer(
            [text, *descriptions],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.inference_mode():
            tokens = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embeddings = functional.normalize(pooled, p=2, dim=1)
            scores = (embeddings[1:] @ embeddings[0]).clamp(-1.0, 1.0)
        return [float(score) for score in scores.tolist()]
