from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from t2i_framework.core.types import AttackCandidate


class Attack(ABC):
    """Base interface for prompt attack modules."""

    name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[AttackCandidate]:
        ...
