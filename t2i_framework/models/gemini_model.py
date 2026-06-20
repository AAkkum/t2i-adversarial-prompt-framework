from __future__ import annotations

from pathlib import Path
from typing import Any

from t2i_framework.core.types import GenerationResult
from t2i_framework.models.base import ImageModel


class GeminiImageModel(ImageModel):
    """Stub for a future Gemini/Imagen black-box API adapter."""

    name = "gemini"

    def generate(
        self,
        prompt: str,
        output_dir: Path,
        seed: int,
        context: dict[str, Any] | None = None,
    ) -> GenerationResult:
        raise NotImplementedError(
            "GeminiImageModel is a stub. Implement API authentication, request submission, "
            "response parsing, and terms-of-service checks before using this adapter."
        )
