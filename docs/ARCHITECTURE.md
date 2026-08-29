# Architecture

The pipeline is:

```text
original prompt
  -> Attack module
  -> Defense module, pre-generation
  -> Image model adapter
  -> Defense module, post-generation
  -> Evaluator / logger
  -> saved image + JSON/CSV result rows
```

The CLI selects one registered model, attack, and defense by name. `ExperimentRunner` coordinates the selected components and writes outputs through `ResultWriter`.

Core extension points:

- `t2i_framework/models/`: the CPU-only mock model and local Hugging Face Diffusers adapters
- `t2i_framework/attacks/`: prompt rewriting or search strategies
- `t2i_framework/defenses/`: prompt filters, image filters, or composite defenses
- `t2i_framework/evaluation/`: result writing and metrics

The mock model is intentionally offline and CPU-only. It renders the selected prompt into a placeholder PNG so the framework can be tested without GPU or model downloads.

The Diffusers adapter loads Hugging Face model pipelines lazily at generation time. The first run of a real model, such as SDXL, may download model weights into the Hugging Face cache before producing an image.

Optional CLIP image-text evaluation runs after image generation. It compares the generated image to the `target_concept` text and stores the cosine similarity in `scores.image_clip_similarity`. When enabled, the `success` field uses the configured CLIP threshold instead of only checking that an image file exists.
