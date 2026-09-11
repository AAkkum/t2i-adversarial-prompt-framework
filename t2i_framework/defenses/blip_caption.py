"""Lazy CPU captioning used only by the image stage of the defense."""

from __future__ import annotations

from pathlib import Path
from typing import Any

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"


class BlipCaptioner:
    """Describe a generated image with BLIP without occupying CUDA VRAM."""

    def __init__(self) -> None:
        self._processor: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is None:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self._processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
            self._model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID).to("cpu")
            self._model.eval()

    def caption(self, image_path: Path) -> str:
        import torch
        from PIL import Image

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image for BLIP does not exist: {path}")
        self._load()
        with Image.open(path) as source:
            image = source.convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=40)
        caption = self._processor.decode(output[0], skip_special_tokens=True).strip()
        if not caption:
            raise RuntimeError("BLIP returned an empty caption.")
        return caption
