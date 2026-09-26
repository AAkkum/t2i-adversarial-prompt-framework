# Components

## Attacks

### `daca`

Uses the paper's ontology-guided Decomposer, Polisher, and Assembler agents to
rewrite a scene as benign descriptions of individual visual components. All
agents use the shared local LLM. It provides a paper-algorithm mode that retains
all edge results and an official-release mode that reproduces the authors'
released orchestration. See `docs/DACA.md`.

### `groot`

Adaptive tree-based attack. It first tests the original prompt, then uses a
local LLM to describe the scene as smaller visual properties. It can also place
the requested scene beside harmless scenes in a multi-panel image. See
`docs/GROOT.md`.

### `pgj`

Uses a local Hugging Face language model to rewrite protected terms as visual
descriptions. This is Abdel's PGJ-based attack.

### `search_attack`

Builds several deterministic prompt variants from phrase lists and concept
replacements. This is Burak's project-specific attack. See
`docs/search_attack.md`.

### `textfooler_style`

Replaces important prompt units with similar alternatives while trying to
preserve the prompt meaning. This is Hans's adaptation of TextFooler for
text-to-image prompts. Its paraphraser and optional semantic fallback judge use
the same local LLM server configured for Groot and the evaluator.

### `identity`

Control that sends the original prompt unchanged.

## Defenses

### `character_filter`

Burak's multi-stage defense:

1. direct blocked-term check;
2. MiniLM semantic prompt check;
3. BLIP caption of the generated image;
4. MiniLM check of that caption.

### `latent_guard_lite`

Hans's lightweight LatentGuard adapter. It compares prompt embeddings with
protected concepts through the released embedding-mapping architecture. It is
not a complete retraining of LatentGuard.

### `safree`

Atabey's full three-stage SDXL port of SAFREE. It detects concept-driving prompt
tokens, projects their second SDXL text-encoder features, chooses the filtering
duration from the prompt itself, and applies Fourier-domain latent re-attention
during generation. See `docs/SAFREE.md` for the exact source/deviation map.

### `trasce`

Generation-time Stable Diffusion 1.4 defense based on the authors' released
TraSCE loss and denoising procedure. See `docs/trasce.md`.

### `none`

Control that allows every prompt and image.

## Evaluator

`LLMImageJudge` is shared by every real-model experiment. It sends the generated
image, original prompt, attacked prompt, target, and policy to the local
multimodal server.

Available success rules:

- `target_presence`: target visible and original intent preserved
- `policy_violation`: policy violation visible and original intent preserved
- `policy_and_target`: all three conditions are true

Every rule also requires the configured confidence threshold.

## Models

- `mock`: placeholder image for fast pipeline tests
- `diffusers`: SDXL, SD 3.5 Medium, SD 3.5 Large, or FLUX

Model presets are under `configs/models/`.
