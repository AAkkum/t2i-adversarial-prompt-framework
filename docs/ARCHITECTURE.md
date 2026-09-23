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

Most attacks return a fixed candidate list. An attack with `adaptive = True`
may inspect each completed result through `process_result(...)` and request the
next candidate through `next_candidate(...)`. The candidate then re-enters the
same defense/model/evaluation path. Groot uses this loop for stage-aware
semantic decomposition and drowning; other attacks retain fixed-list behavior.

The CLI selects one registered model, attack, and defense by name. `ExperimentRunner` coordinates the selected components and writes outputs through `ResultWriter`.

Core extension points:

- `t2i_framework/models/`: the CPU-only mock model and local Hugging Face Diffusers adapters
- `t2i_framework/attacks/`: prompt rewriting or search strategies
- `t2i_framework/defenses/`: prompt filters, image filters, or composite defenses
- `t2i_framework/evaluation/`: result writing and metrics

The mock model is intentionally offline and CPU-only. It renders the selected prompt into a placeholder PNG so the framework can be tested without GPU or model downloads.

The Diffusers adapter loads Hugging Face model pipelines lazily at generation time. The first run of a real model, such as SDXL, may download model weights into the Hugging Face cache before producing an image.

Optional CLIP image-text evaluation runs after image generation. It compares the generated image to the `target_concept` text and stores the cosine similarity in `scores.image_clip_similarity`. When enabled, the `success` field uses the configured CLIP threshold instead of only checking that an image file exists.

Optional prompt-prompt similarity compares the original prompt text with the attacked prompt text and stores the result in `scores.prompt_prompt_similarity`. This is an auxiliary drift metric for prompt rewriting attacks; it does not decide whether the generated image preserved the target concept.

For prompt-file runs, `read_prompt_file` accepts CSV, JSON, and JSONL. The runner keeps one model adapter instance for the whole prompt list, so Diffusers pipelines can stay loaded across a batch instead of being reloaded per prompt.

Generated images are first written into a temporary quarantine directory. If the image-stage defense allows the image, it is published into the output directory. If the image-stage defense blocks or errors, the generated image is not retained as a final result.
