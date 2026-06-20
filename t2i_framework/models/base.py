from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from t2i_framework.core.types import GenerationResult


class ImageModel(ABC):
    """Base interface for image generation adapters."""

    name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        output_dir: Path,
        seed: int,
        context: dict[str, Any] | None = None,
    ) -> GenerationResult:
        ...
