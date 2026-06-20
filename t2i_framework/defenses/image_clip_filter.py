from __future__ import annotations

from pathlib import Path
from typing import Any

from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense


class ImageClipFilterDefense(Defense):
    """Placeholder for future CLIP-based image-level concept filtering."""

    name = "image_clip_filter"

    def check_image(
        self,
        image_path: Path,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        return DefenseDecision(
            allowed=True,
            reason="image CLIP filter placeholder; install optional model dependencies later",
            score=None,
        )
