from __future__ import annotations

import textwrap
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from t2i_framework.core.types import GenerationResult
from t2i_framework.models.base import ImageModel


class MockImageModel(ImageModel):
    """CPU-only placeholder model that writes prompt text onto a PNG."""

    name = "mock"

    def generate(
        self,
        prompt: str,
        output_dir: Path,
        seed: int,
        context: dict[str, Any] | None = None,
    ) -> GenerationResult:
        context = context or {}
        run_id = context.get("run_id") or f"seed_{seed}_{uuid.uuid4().hex[:8]}"
        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        filename = context.get("output_filename", f"{run_id}.png")
        image_path = _available_path(image_dir / filename)

        image = Image.new("RGB", (768, 512), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        lines = [
            "Model: mock",
            f"Seed: {seed}",
            "",
            "Prompt:",
            *textwrap.wrap(prompt, width=72),
        ]
        y = 32
        for line in lines:
            draw.text((32, y), line, fill="black", font=font)
            y += 18
        image.save(image_path)

        return GenerationResult(
            prompt=prompt,
            image_path=image_path,
            seed=seed,
            model_name=self.name,
            metadata={"run_id": run_id, "kind": "placeholder"},
        )


def _available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused filename for {path}")
