# Components

## Attacks

### `daca`

Uses ontology-guided Decomposer, Polisher, and Assembler agents to rewrite a
scene as descriptions of its visual parts. The agents use the local LLM. See
`docs/DACA.md`.

### `groot`

Starts with the original prompt and then adaptively applies tree-based semantic
decomposition and multi-panel drowning when earlier candidates fail. See
`docs/GROOT.md`.

### `pgj`

Uses a local Hugging Face language model to replace protected terms with visual
descriptions.

### `ring_a_bell`

Uses a frozen Stable Diffusion 1.4 CLIP text encoder and a genetic search to
find token sequences near a target concept embedding. See
`docs/ring_a_bell.md`.

### `identity`

Control that sends the original prompt unchanged.

## Defenses

### `latent_guard_lite`

Prompt-stage adapter around the released Latent Guard embedding-mapping
architecture and weights. It blocks prompts whose mapped embedding is close to
a configured protected concept.

### `safree`

Three-stage SDXL defense: prompt-token projection, self-validation, and latent
re-attention during generation. See `docs/SAFREE.md`.

### `trasce`

Stable Diffusion 1.4 generation defense that steers the diffusion trajectory
away from the target concept. See `docs/trasce.md`.

### `none`

Control that allows every prompt and image.

## Evaluator

`LLMImageJudge` reviews every generated image after the attack and defense have
finished. It receives the original prompt, attacked prompt, target, and policy.

Success modes:

- `target_presence`: target visible and original intent preserved
- `policy_violation`: policy violation visible and original intent preserved
- `policy_and_target`: target, policy violation, and intent preservation all true

The configured confidence threshold must also be met. Evaluator success and
defense blocking are stored separately.

## Models

- `mock`: placeholder image for fast pipeline tests
- `diffusers`: model adapter used by SDXL, SD 1.4, SD 3.5, and FLUX presets

Model presets are under `configs/models/`.
