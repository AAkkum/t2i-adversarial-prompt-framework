from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from t2i_framework.core.logging_utils import console
from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense
from t2i_framework.similarity.clip_text import CLIPTextSimilarityScorer


SimilarityScorer = Callable[[str, list[str]], list[float]]


class CLIPSimilarityDefense(Defense):
    """Pre-generation prompt defense using CLIP text similarity to restricted concepts."""

    name = "clip_similarity"

    def __init__(
        self,
        concepts_path: Path | None = None,
        threshold: float = 0.7,
        include_aliases: bool = True,
        log_similarity: bool = True,
        similarity_scorer: SimilarityScorer | None = None,
    ) -> None:
        self.concepts_path = concepts_path or Path("data/restricted_concepts.yaml")
        self.threshold = threshold
        self.include_aliases = include_aliases
        self.log_similarity = log_similarity
        self.similarity_scorer = similarity_scorer
        self.concepts = self._load_concepts(self.concepts_path)
        self._clip_similarity_scorer = None
        self._similarity_warning_printed = False

    def check_prompt(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        context = context or {}
        self._apply_context_config(context)
        terms = self._restricted_terms(target_concept)
        if not terms:
            return DefenseDecision(
                allowed=True,
                reason="no restricted concepts configured",
                score=0.0,
                metadata={"threshold": self.threshold, "checked_terms": 0},
            )

        scores = self._score_many(prompt, terms)
        if not scores:
            return DefenseDecision(
                allowed=True,
                reason="CLIP similarity unavailable",
                score=None,
                metadata={"threshold": self.threshold, "checked_terms": len(terms)},
            )

        best_index, best_score = max(enumerate(scores), key=lambda item: item[1])
        matched_term = terms[best_index]
        allowed = best_score < self.threshold
        reason = (
            f"CLIP similarity {best_score:.3f} >= threshold {self.threshold:.3f}: {matched_term}"
            if not allowed
            else f"max CLIP similarity {best_score:.3f} below threshold {self.threshold:.3f}"
        )

        if self.log_similarity:
            console.print(
                "[clip_similarity] "
                f'prompt="{prompt}" matched="{matched_term}" '
                f"similarity={best_score:.3f} threshold={self.threshold:.3f} "
                f"allowed={allowed}",
                markup=False,
            )

        return DefenseDecision(
            allowed=allowed,
            reason=reason,
            score=best_score,
            metadata={
                "matched_term": matched_term,
                "similarity": best_score,
                "threshold": self.threshold,
                "checked_terms": len(terms),
            },
        )

    @staticmethod
    def _load_concepts(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data.get("concepts", {})

    def _restricted_terms(self, target_concept: str | None) -> list[str]:
        terms: list[str] = []
        if target_concept:
            terms.append(target_concept)

        for concept, config in self.concepts.items():
            terms.append(concept)
            if self.include_aliases:
                terms.extend(config.get("aliases", []))

        unique_terms = []
        seen = set()
        for term in terms:
            key = str(term).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique_terms.append(str(term).strip())
        return unique_terms

    def _score_many(self, prompt: str, terms: list[str]) -> list[float]:
        if self.similarity_scorer is not None:
            return [float(value) for value in self.similarity_scorer(prompt, terms)]

        try:
            if self._clip_similarity_scorer is None:
                self._clip_similarity_scorer = CLIPTextSimilarityScorer()
            return self._clip_similarity_scorer.score_many(prompt, terms)
        except RuntimeError as exc:
            if not self._similarity_warning_printed:
                console.print(f"[clip_similarity] CLIP similarity disabled: {exc}")
                self._similarity_warning_printed = True
            return []

    def _apply_context_config(self, context: dict[str, Any]) -> None:
        defense_config = dict((context.get("config") or {}).get("defense", {}))
        defense_config.pop("name", None)

        if "concepts_path" in defense_config:
            concepts_path = Path(defense_config["concepts_path"])
            if concepts_path != self.concepts_path:
                self.concepts_path = concepts_path
                self.concepts = self._load_concepts(self.concepts_path)

        for key in ["threshold", "include_aliases", "log_similarity"]:
            if key in defense_config:
                setattr(self, key, defense_config[key])
