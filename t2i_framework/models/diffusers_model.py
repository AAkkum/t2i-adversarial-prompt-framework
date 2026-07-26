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
        enable_model_cpu_offload: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self._pipeline: Any | None = None

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
        generator_device = "cpu" if self.enable_model_cpu_offload else device
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        if self._pipeline is None:
            self._pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_id, torch_dtype=torch_dtype
            )
            if self.enable_model_cpu_offload:
                if device != "cuda":
                    raise RuntimeError("Model CPU offloading requires CUDA.")
                self._pipeline.enable_model_cpu_offload(device=device)
            else:
                self._pipeline = self._pipeline.to(device)
        result = self._pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            width=self.width,
            height=self.height,
        )

        context = context or {}
        image_dir = output_dir if context.get("output_filename") else output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        run_id = context.get("run_id", f"seed_{seed}")
        filename = context.get("output_filename", f"{run_id}.png")
        image_path = _available_path(image_dir / filename)
        result.images[0].save(image_path)
        return GenerationResult(
            prompt=prompt,
            image_path=image_path,
            seed=seed,
            model_name=self.name,
            metadata={
                "model_id": self.model_id,
                "device": device,
                "dtype": dtype_name,
                "model_cpu_offload": self.enable_model_cpu_offload,
                "width": self.width,
                "height": self.height,
                "num_inference_steps": self.num_inference_steps,
            },
        )


def _available_path(path: Path) -> Path:
    """Avoid overwriting an existing image."""

    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused filename for {path}")
