# T2I Adversarial Prompt Framework

University project for testing prompt attacks and defenses against local
text-to-image models.

## Components

Main attacks:

- `daca`
- `groot`
- `pgj`
- `ring_a_bell`
- `search_attack`
- `textfooler_style`

Main defenses:

- `character_filter`
- `latent_guard_lite`
- `safree` (Stable Diffusion XL)
- `trasce` (Stable Diffusion 1.4)

`identity` and `none` are controls. One shared local multimodal LLM evaluates
the generated images. CLIP is not used as the experiment evaluator.

Ring-A-Bell and TraSCE use `configs/models/sd14.yaml` for the published-method
experiment. See [Ring-A-Bell](docs/ring_a_bell.md) and [TraSCE](docs/trasce.md) for
commands, defaults, project-created concept pairs and documented deviations.
SAFREE uses the normal `configs/models/sdxl.yaml` preset; its complete source
and deviation map is in [SAFREE](docs/SAFREE.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[full]"
```

The Diffusers models may require Hugging Face login and accepted model terms:

```bash
huggingface-cli login
```

## Local LLM

Groot, DACA, and TextFooler use a local LLM to rewrite prompts. The same server
evaluates generated images for every real-model run.

```bash
scripts/start-local-llm.sh
```

The model and server settings are in `configs/local_llm.yaml`. If its `model`
value is a Hugging Face llama.cpp model spec, the start command downloads and
caches it automatically when needed. The default server is
the host and port configured in `configs/local_llm.yaml`, with API alias
`local-llm`.

## Quick Checks

The mock model checks the pipeline without loading learned models. It does not
run the image evaluator because its output is only a placeholder image.

```bash
python main.py \
  --model mock \
  --attack identity \
  --defense none \
  --prompt "a blue rabbit mascot standing in a garden" \
  --target "blue rabbit mascot"
```

Real SDXL run:

```bash
python main.py \
  --model diffusers \
  --model-config configs/models/sdxl.yaml \
  --attack groot \
  --defense none \
  --prompt "a blue rabbit mascot standing in a garden" \
  --target "blue rabbit mascot" \
  --max-candidates 5
```

Groot against the three-stage SDXL SAFREE defense:

```bash
python main.py \
  --model diffusers \
  --model-config configs/models/sdxl.yaml \
  --attack groot \
  --defense safree \
  --prompt "a blue rabbit mascot standing in a garden" \
  --target "blue rabbit mascot" \
  --max-candidates 3
```

Real Diffusers runs automatically load `configs/evaluation/llm_judge.yaml`.
Its `policy_violation` rule asks whether unsafe content remains visible and the
original scene is preserved. Change it to `target_presence` only for benign
target-preservation experiments.

Batch run:

```bash
python main.py \
  --model diffusers \
  --model-config configs/models/sdxl.yaml \
  --attack groot \
  --defense character_filter \
  --prompt-file data/datasets/safety_nonsexual/safety_nonsexual_100.csv \
  --max-candidates 3
```

Start with `data/datasets/safety_nonsexual/safety_nonsexual_10.csv` before an
expensive 100-row run. The separate
`data/datasets/representative/representative_prompt_batch_100.csv` remains the
benign target-preservation benchmark and should use `target_presence`.

## Results

Each run writes:

- `results.jsonl`: compact result, one line per candidate
- `results.csv`: the same compact data for spreadsheets
- `details.jsonl`: full technical trace for debugging
- `config.yaml`: merged run configuration
- `images/`: allowed generated images

`success` is decided by the shared LLM evaluator. Defense decisions are shown
separately as `prompt_blocked`, `image_blocked`, and `defense_bypassed`.

## Documentation

- `docs/COMPONENTS.md`: current attacks, defenses, and evaluator
- `docs/ARCHITECTURE.md`: short pipeline description
- `docs/GROOT.md`: Groot behavior and local server setup
- `docs/SAFREE.md`: SDXL SAFREE stages, source map, deviations, and command
- `docs/search_attack.md`: Search Attack behavior
- `docs/PAPER_IMPLEMENTATION_MAP.md`: owners, papers, and deviations
- `docs/HOW_TO_ADD_DEFENSE.md`: minimal defense implementation steps
- `docs/SAFETY_SCOPE.md`: project safety boundary
