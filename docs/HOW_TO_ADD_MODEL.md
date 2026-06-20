# How To Add A Model

Models implement `ImageModel.generate(...)` and return a `GenerationResult`.

Local Hugging Face models should use lazy imports so that the default mock pipeline does not require `torch`, `diffusers`, or `transformers`. See `DiffusersImageModel` for the pattern.

This project is scoped to local/open Hugging Face `diffusers` or `transformers` models.

Current Diffusers config files:

- `configs/sdxl.yaml`: `stabilityai/stable-diffusion-xl-base-1.0`
- `configs/sd35_medium.yaml`: `stabilityai/stable-diffusion-3.5-medium`
- `configs/flux.yaml`: `black-forest-labs/FLUX.1-schnell`

After creating a model class, register it in `t2i_framework/core/registry.py` under `MODEL_REGISTRY`.
