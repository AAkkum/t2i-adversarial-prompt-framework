# How To Add A Model

Models implement `ImageModel.generate(...)` and return a `GenerationResult`.

Local Hugging Face models should use lazy imports so that the default mock pipeline does not require `torch`, `diffusers`, or `transformers`. See `DiffusersImageModel` for the pattern.

Black-box API models, such as Gemini/Imagen, should isolate authentication, request submission, and response parsing inside their adapter. Only test closed-source APIs when the supervisor has approved the experiment and the provider terms allow it.

After creating a model class, register it in `t2i_framework/core/registry.py` under `MODEL_REGISTRY`.
