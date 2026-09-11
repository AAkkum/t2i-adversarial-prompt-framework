from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense
from t2i_framework.defenses.blip_caption import BlipCaptioner
from t2i_framework.defenses.semantic_concepts import MiniLMConceptMatcher, SemanticMatch

DATA_DIR = Path(__file__).resolve().parents[2] / "data/search_attack"

# Thresholds for the local experiments; results depend on the test prompts.
SEMANTIC_PROMPT_THRESHOLD = 0.50
SEMANTIC_IMAGE_THRESHOLD = 0.50


def normalize_character_text(text: str) -> str:
    """Normalize case, Unicode representation, and whitespace for matching."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def load_blocked_terms(path: Path) -> tuple[str, ...]:
    """Load unique, normalized whole terms from an editable UTF-8 text file."""

    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    terms = {
        normalize_character_text(line)
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }
    return tuple(sorted(terms, key=lambda term: (-len(term), term)))


def load_protected_concepts(path: Path) -> dict[str, str]:
    """Load normalized block-term to protected-description mappings from JSON."""

    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise TypeError("concept_targets.json must contain a JSON object.")
    concepts: dict[str, str] = {}
    for term, description in raw.items():
        if not isinstance(term, str) or not term.strip():
            raise ValueError("Every protected concept key must be non-empty text.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Every protected concept description must be non-empty text.")
        concepts[normalize_character_text(term)] = description.strip()
    return concepts


class CharacterFilterDefense(Defense):
    """Keyword, MiniLM prompt, and BLIP/MiniLM image defense for local tests."""

    name = "character_filter"

    def __init__(
        self,
        terms_path: Path | None = None,
        concepts_path: Path | None = None,
        *,
        enable_semantic_prompt: bool = True,
        enable_image_semantic: bool = True,
        semantic_prompt_threshold: float = SEMANTIC_PROMPT_THRESHOLD,
        semantic_image_threshold: float = SEMANTIC_IMAGE_THRESHOLD,
        semantic_matcher: MiniLMConceptMatcher | None = None,
        captioner: BlipCaptioner | None = None,
    ) -> None:
        self.terms_path = Path(terms_path) if terms_path else DATA_DIR / "blocked_terms.txt"
        self.concepts_path = (
            Path(concepts_path) if concepts_path else DATA_DIR / "concept_targets.json"
        )
        self.blocked_terms = load_blocked_terms(self.terms_path)
        self.protected_concepts = load_protected_concepts(self.concepts_path)
        self.enable_semantic_prompt = enable_semantic_prompt
        self.enable_image_semantic = enable_image_semantic
        self.semantic_prompt_threshold = _validate_threshold(
            semantic_prompt_threshold, "semantic_prompt_threshold"
        )
        self.semantic_image_threshold = _validate_threshold(
            semantic_image_threshold, "semantic_image_threshold"
        )
        self.semantic_matcher = semantic_matcher or MiniLMConceptMatcher()
        self.captioner = captioner or BlipCaptioner()

    def check_prompt(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        self._apply_context_config(context)
        del target_concept  # Protected concepts come only from concept_targets.json.
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("character_filter requires a non-empty prompt.")

        matched_term = self._find_keyword(prompt)
        if matched_term is not None:
            concept = self.protected_concepts.get(matched_term)
            metadata = {
                "direct_blocked_term": matched_term,
                "matched_keyword_block": matched_term,
                "closest_protected_concept": concept,
                "keyword_result": "BLOCKED",
                "semantic_prompt_result": "NOT_RUN",
                "semantic_prompt_similarity": None,
                "semantic_prompt_threshold": self.semantic_prompt_threshold,
                "blocked_by": "KEYWORD",
                "final_defense": "BLOCKED",
            }
            decision = DefenseDecision(
                allowed=False,
                reason=f"Blocked term: {matched_term}",
                score=1.0,
                metadata=metadata,
            )
            self._print_prompt(context, prompt, decision)
            return decision

        if not self.enable_semantic_prompt:
            decision = DefenseDecision(
                allowed=True,
                reason="Keyword defense passed; MiniLM prompt defense disabled",
                score=0.0,
                metadata={
                    "direct_blocked_term": None,
                    "matched_keyword_block": None,
                    "closest_protected_concept": None,
                    "keyword_result": "PASS",
                    "semantic_prompt_result": "DISABLED",
                    "semantic_prompt_similarity": None,
                    "semantic_prompt_threshold": self.semantic_prompt_threshold,
                    "blocked_by": "NONE",
                    "final_defense": "ALLOWED",
                },
            )
            self._print_prompt(context, prompt, decision)
            return decision

        match = self.semantic_matcher.match(prompt, self.protected_concepts)
        blocked = match.similarity >= self.semantic_prompt_threshold
        metadata = self._semantic_prompt_metadata(match, blocked)
        decision = DefenseDecision(
            allowed=not blocked,
            reason=(
                f"Semantic prompt matched protected concept: {match.protected_concept}"
                if blocked
                else "Keyword and MiniLM prompt defenses passed"
            ),
            score=match.similarity,
            metadata=metadata,
        )
        self._print_prompt(context, prompt, decision)
        return decision

    def check_image(
        self,
        image_path: Path,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        self._apply_context_config(context)
        del target_concept
        if not self.enable_image_semantic:
            return DefenseDecision(
                allowed=True,
                reason="BLIP/MiniLM image defense disabled",
                metadata={"image_result": "DISABLED", "blocked_by": "NONE"},
            )

        if context is not None:
            context["image_error_stage"] = "blip"
        caption = self.captioner.caption(Path(image_path))
        if context is not None:
            context["blip_caption"] = caption
            context["image_error_stage"] = "minilm_image"
        print("BLIP executed: YES")
        print(f"BLIP Caption: {caption}")
        match = self.semantic_matcher.match(caption, self.protected_concepts)
        blocked = match.similarity >= self.semantic_image_threshold
        metadata = {
            "blip_caption": caption,
            "closest_image_blocked_term": match.blocked_term,
            "closest_image_protected_concept": match.protected_concept,
            "semantic_image_similarity": match.similarity,
            "semantic_image_threshold": self.semantic_image_threshold,
            "image_result": "BLOCKED" if blocked else "PASS",
            "blocked_by": "BLIP_MINILM_IMAGE" if blocked else "NONE",
            "final_defense": "BLOCKED" if blocked else "ALLOWED",
        }
        decision = DefenseDecision(
            allowed=not blocked,
            reason=(
                f"Image caption matched protected concept: {match.protected_concept}"
                if blocked
                else "BLIP/MiniLM image defense passed"
            ),
            score=match.similarity,
            metadata=metadata,
        )
        self._print_image(context, Path(image_path), decision)
        return decision

    def _apply_context_config(self, context: dict[str, Any] | None) -> None:
        defense_config = dict(((context or {}).get("config") or {}).get("defense", {}))
        defense_config.pop("name", None)

        terms_path = defense_config.get("terms_path")
        if terms_path is not None and Path(terms_path) != self.terms_path:
            self.terms_path = Path(terms_path)
            self.blocked_terms = load_blocked_terms(self.terms_path)

        concepts_path = defense_config.get("concepts_path")
        if concepts_path is not None and Path(concepts_path) != self.concepts_path:
            self.concepts_path = Path(concepts_path)
            self.protected_concepts = load_protected_concepts(self.concepts_path)

        if "enable_semantic_prompt" in defense_config:
            self.enable_semantic_prompt = bool(defense_config["enable_semantic_prompt"])
        if "enable_image_semantic" in defense_config:
            self.enable_image_semantic = bool(defense_config["enable_image_semantic"])
        if "semantic_prompt_threshold" in defense_config:
            self.semantic_prompt_threshold = _validate_threshold(
                defense_config["semantic_prompt_threshold"],
                "semantic_prompt_threshold",
            )
        if "semantic_image_threshold" in defense_config:
            self.semantic_image_threshold = _validate_threshold(
                defense_config["semantic_image_threshold"],
                "semantic_image_threshold",
            )

    def _find_keyword(self, prompt: str) -> str | None:
        normalized_prompt = normalize_character_text(prompt)
        for term in self.blocked_terms:
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized_prompt):
                return term
        return None

    def _semantic_prompt_metadata(
        self, match: SemanticMatch, blocked: bool
    ) -> dict[str, Any]:
        return {
            "direct_blocked_term": None,
            "matched_keyword_block": None,
            "closest_blocked_term": match.blocked_term,
            "closest_protected_concept": match.protected_concept,
            "keyword_result": "PASS",
            "semantic_prompt_result": "BLOCKED" if blocked else "PASS",
            "semantic_prompt_similarity": match.similarity,
            "semantic_prompt_threshold": self.semantic_prompt_threshold,
            "blocked_by": "MINILM_PROMPT" if blocked else "NONE",
            "final_defense": "BLOCKED" if blocked else "ALLOWED",
        }

    @staticmethod
    def _print_prompt(
        context: dict[str, Any] | None, prompt: str, decision: DefenseDecision
    ) -> None:
        data = decision.metadata
        candidate = (context or {}).get("candidate_index", "?")
        print(f"\nDefense result for Candidate {candidate}")
        print(f"Prompt: {prompt}")
        print(f"Direct blocked term: {data.get('direct_blocked_term') or 'NONE'}")
        print(f"Matched keyword block: {data.get('matched_keyword_block') or 'NONE'}")
        print(f"Closest protected concept: {data.get('closest_protected_concept') or 'NONE'}")
        print(f"Keyword result: {data['keyword_result']}")
        similarity = data.get("semantic_prompt_similarity")
        print(
            "MiniLM prompt similarity: NOT RUN"
            if similarity is None
            else f"MiniLM prompt similarity: {similarity:.4f}"
        )
        print(f"MiniLM threshold: {data['semantic_prompt_threshold']:.4f}")
        print(f"MiniLM result: {data['semantic_prompt_result']}")
        print(f"Blocked By: {data['blocked_by']}")
        print(f"Final Defense: {data['final_defense']}")

    @staticmethod
    def _print_image(
        context: dict[str, Any] | None, image_path: Path, decision: DefenseDecision
    ) -> None:
        data = decision.metadata
        candidate = (context or {}).get("candidate_index", "?")
        print(f"\nImage defense for Candidate {candidate}")
        print(f"Closest image protected concept: {data['closest_image_protected_concept']}")
        print(f"MiniLM image similarity: {data['semantic_image_similarity']:.4f}")
        print(f"MiniLM image threshold: {data['semantic_image_threshold']:.4f}")
        print(f"MiniLM image result: {data['image_result']}")
        print(f"Blocked By: {data['blocked_by']}")
        print(f"Final Defense: {data['final_defense']}")


def _validate_threshold(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number between 0 and 1.")
    return float(value)
