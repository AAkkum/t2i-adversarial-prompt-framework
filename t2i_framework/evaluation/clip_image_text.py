from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image


class ImageTextScorer(Protocol):
    """Interface for image-text similarity scorers."""

    def score(self, image_path: Path, text: str) -> float:
        ...


class CLIPImageTextScorer:
    """Lazy CLIP image-text cosine similarity scorer."""

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None

    def score(self, image_path: Path, text: str) -> float:
        self._load()
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._torch.inference_mode():
            image_features = self._model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = self._model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            image_features = self._extract_tensor(image_features)
            text_features = self._extract_tensor(text_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features[0] * text_features[0]).sum().item()

        return float(similarity)

    def _extract_tensor(self, features: object):
        if isinstance(features, self._torch.Tensor):
            return features

        for attr_name in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(features, attr_name, None)
            if isinstance(value, self._torch.Tensor):
                return value

        if isinstance(features, (tuple, list)):
            for value in features:
                if isinstance(value, self._torch.Tensor) and value.ndim == 2:
                    return value

        raise RuntimeError(
            "CLIP image-text scorer could not read embeddings from the installed "
            "transformers CLIP output."
        )

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLIP image-text scoring requires torch and transformers. "
                'Install model dependencies with: pip install -e ".[models]"'
            ) from exc

        self._torch = torch
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._processor = CLIPProcessor.from_pretrained(self.model_id)
            self._model = CLIPModel.from_pretrained(self.model_id).to(self._device)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                "CLIP image-text scoring could not load the CLIP model. "
                "Run once with internet access so Hugging Face can cache "
                f"{self.model_id}, or disable CLIP evaluation in the config."
            ) from exc
        self._model.eval()
