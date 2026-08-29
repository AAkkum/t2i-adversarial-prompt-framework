from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml

from t2i_framework.attacks.base import Attack
from t2i_framework.core.types import AttackCandidate


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _replace_phrase(prompt: str, phrase: str, replacement: str) -> str | None:
    pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
    rewritten, count = pattern.subn(replacement, prompt, count=1)
    return rewritten if count else None


class GrootLiteAttack(Attack):
    """Rule-based semantic decomposition attack for safe synthetic concepts."""

    name = "groot_lite"

    def __init__(self, decompositions_path: Path | None = None) -> None:
        self.decompositions_path = decompositions_path or Path("data/groot_decompositions.yaml")
        self.decompositions = self._load_decompositions(self.decompositions_path)

    @staticmethod
    def _load_decompositions(path: Path) -> dict[str, tuple[str, ...]]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        decompositions: dict[str, tuple[str, ...]] = {}
        for concept, config in data.get("concepts", {}).items():
            values = config.get("decompositions", [])
            if isinstance(values, list):
                clean_values = tuple(str(value).strip() for value in values if str(value).strip())
                if clean_values:
                    decompositions[_normalize(str(concept))] = clean_values
        return decompositions

    def generate(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[AttackCandidate]:
        context = context or {}
        self._apply_context_config(context)

        if not target_concept:
            return [
                AttackCandidate(
                    text=prompt,
                    metadata={
                        "method": self.name,
                        "status": "no_target_concept",
                    },
                )
            ]

        normalized_target = _normalize(target_concept)
        replacements = self.decompositions.get(normalized_target)
        if not replacements:
            return [
                AttackCandidate(
                    text=prompt,
                    metadata={
                        "method": self.name,
                        "status": "unsupported_target_concept",
                        "target_concept": target_concept,
                        "decompositions_path": str(self.decompositions_path),
                    },
                )
            ]

        candidates: list[AttackCandidate] = []
        for replacement in replacements:
            rewritten = _replace_phrase(prompt, target_concept, replacement)
            if rewritten is None:
                rewritten = f"{prompt}, visual attributes: {replacement}"
                rewrite_mode = "attribute_append"
            else:
                rewrite_mode = "target_replacement"

            candidates.append(
                AttackCandidate(
                    text=rewritten,
                    metadata={
                        "method": self.name,
                        "strategy": "semantic_decomposition",
                        "rewrite_mode": rewrite_mode,
                        "target_concept": target_concept,
                        "replacement": replacement,
                        "decompositions_path": str(self.decompositions_path),
                    },
                )
            )

        return candidates

    def _apply_context_config(self, context: dict[str, Any]) -> None:
        attack_config = dict((context.get("config") or {}).get("attack", {}))
        nested_config = attack_config.get(self.name)
        if isinstance(nested_config, dict):
            attack_config.update(nested_config)

        path_value = attack_config.get("decompositions_path")
        if not path_value:
            return

        configured_path = Path(path_value)
        if configured_path == self.decompositions_path:
            return

        self.decompositions_path = configured_path
        self.decompositions = self._load_decompositions(self.decompositions_path)
