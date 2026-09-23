# Groot

This attack follows the workflow from *Groot: Adversarial Testing for
Text-to-Image Generative Models with Tree-based Semantic Transformation*
(arXiv:2402.12100). The authors published their code as
[TREANT](https://github.com/llm-jailbreaker/Treant).

## How It Works

For each input prompt:

1. Test the original prompt.
2. If the prompt is blocked, ask the local LLM to split the scene into visible
   objects, properties, and relationships.
3. Turn those descriptions into a new prompt.
4. If the generated image is blocked or the evaluator reports failure, create
   a multi-panel prompt containing the requested scene and harmless scenes.
5. Continue until success or `--max-candidates` is reached.

`tree.max_depth` and `tree.max_nodes` limit decomposition. `drowning.panel_count`
controls the number of image panels. The `full`, `semantic_only`, and
`drowning_only` modes reproduce the main method and its ablations.

## Local Server

Choose the model and server settings in `configs/local_llm.yaml`, then start it:

```bash
scripts/start-local-llm.sh
```

When `local_llm.model` contains a Hugging Face llama.cpp model spec, llama.cpp
downloads and caches it automatically if it is missing. A local `.gguf` path is
also accepted. The server listens on port `8082` by default. Groot uses it for
prompt-tree construction, and the evaluator uses the same server for image
review.

The LLM and image model remain loaded at the same time. If VRAM is insufficient,
lower `local_llm.gpu_layers` so llama.cpp keeps more layers in system
RAM. SDXL uses `device_map: cuda` to load Diffusers components directly onto the
GPU instead of first constructing the full pipeline in system RAM.

## Example

```bash
python main.py \
  --model diffusers \
  --model-config configs/models/sdxl.yaml \
  --attack groot \
  --defense character_filter \
  --prompt "a blue rabbit mascot standing in a garden" \
  --target "blue rabbit mascot" \
  --max-candidates 5
```

The shared evaluator is configured in `configs/evaluation/llm_judge.yaml`.
`target_presence` is appropriate for the harmless representative dataset.
`policy_violation` is appropriate only for a policy-labelled dataset.

## Differences From The Paper

This is a method reimplementation, not an exact result reproduction:

- a local multimodal model replaces GPT-4;
- local Diffusers models replace the paper's hosted model set;
- the shared LLM evaluator replaces manual image labels;
- project datasets replace NSFW-1k;
- framework defenses replace hosted-service error feedback.

These changes make the experiment local and reproducible, but its numerical
results cannot be directly compared with the paper's reported percentages.
