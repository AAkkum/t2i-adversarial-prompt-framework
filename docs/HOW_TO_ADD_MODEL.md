# How To Add A Model

Models implement `ImageModel.generate(...)` and return a `GenerationResult`.

Local Hugging Face models should use lazy imports so that the default mock pipeline does not require `torch`, `diffusers`, or `transformers`. See `DiffusersImageModel` for the pattern.

This project is scoped to local/open Hugging Face `diffusers` or `transformers` models.

Current Diffusers config files:

- `configs/sdxl.yaml`: `stabilityai/stable-diffusion-xl-base-1.0` - smoke-tested locally
- `configs/sd35_medium.yaml`: `stabilityai/stable-diffusion-3.5-medium` - configured, not yet runtime-validated
- `configs/flux.yaml`: `black-forest-labs/FLUX.1-schnell` - configured, not yet runtime-validated

Install model dependencies before using the Diffusers adapter:

```bash
pip install -e ".[models]"
```

The first real model run may download large model files into the Hugging Face cache. The configs only choose which model is loaded; they do not download all configured models automatically.

After creating a model class, register it in `t2i_framework/core/registry.py` under `MODEL_REGISTRY`.
