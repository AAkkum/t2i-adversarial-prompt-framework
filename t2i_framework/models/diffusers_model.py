from __future__ import annotations

from pathlib import Path
from typing import Any

from t2i_framework.core.types import GenerationResult
from t2i_framework.models.base import ImageModel


class DiffusersImageModel(ImageModel):
    """Minimal Hugging Face Diffusers adapter with lazy optional imports."""

    name = "diffusers"

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str | None = None,
        dtype: str | None = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height

    def generate(
        self,
        prompt: str,
        output_dir: Path,
        seed: int,
        context: dict[str, Any] | None = None,
    ) -> GenerationResult:
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise RuntimeError(
                "DiffusersImageModel requires optional dependencies. "
                'Install them with: pip install -e ".[models]"'
            ) from exc

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_name = self.dtype or ("float16" if device == "cuda" else "float32")
        torch_dtype = getattr(torch, dtype_name)
        generator = torch.Generator(device=device).manual_seed(seed)
        pipeline = AutoPipelineForText2Image.from_pretrained(self.model_id, torch_dtype=torch_dtype)
        pipeline = pipeline.to(device)
        result = pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            width=self.width,
            height=self.height,
        )

        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        run_id = (context or {}).get("run_id", f"seed_{seed}")
        image_path = image_dir / f"{run_id}.png"
        result.images[0].save(image_path)
        return GenerationResult(
            prompt=prompt,
            image_path=image_path,
            seed=seed,
            model_name=self.name,
            metadata={"model_id": self.model_id, "device": device, "dtype": dtype_name},
        )
