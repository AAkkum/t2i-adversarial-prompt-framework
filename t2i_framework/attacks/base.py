from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from t2i_framework.core.types import AttackCandidate, EvaluationResult


class Attack(ABC):
    """Base interface for prompt attack modules."""

    name: str
    adaptive: bool = False

    @abstractmethod
    def generate(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[AttackCandidate]:
        ...

    def cleanup(self, context: dict[str, Any] | None = None) -> None:
        """Release optional resources held by an attack."""

        return None

    def process_result(
        self,
        result: EvaluationResult,
        candidate: AttackCandidate,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Optionally add attack-specific evaluation to a completed candidate result."""

        return None

    def next_candidate(
        self,
        previous: AttackCandidate,
        result: EvaluationResult,
        context: dict[str, Any] | None = None,
    ) -> AttackCandidate | None:
        """Return an adaptive follow-up candidate after observing a completed result."""

        return None
