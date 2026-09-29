# T2I Adversarial Prompt Framework

University project for evaluating prompt attacks and defenses on local
text-to-image models.

## Final Components

Attacks:

- `daca`
- `groot`
- `pgj`
- `ring_a_bell`
- `identity` (unchanged-prompt control)

Defenses:

- `latent_guard_lite` (prompt-stage)
- `safree` (generation-stage, SDXL)
- `trasce` (generation-stage, Stable Diffusion 1.4)
- `none` (no-defense control)

Every real-model run uses the same local multimodal LLM evaluator after image
generation. The evaluator is separate from the defenses.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[full]"
```

Some Diffusers models require a Hugging Face account and accepted model terms:

```bash
huggingface-cli login
```

## Local LLM

The shared multimodal server evaluates generated images and also builds Groot
prompt trees.

```bash
scripts/start-local-llm.sh
```

Model and server settings are in `configs/local_llm.yaml`. A Hugging Face
llama.cpp model specification is downloaded and cached automatically when first
started. DACA uses a separate text-only server started with
`scripts/start-daca-llm.sh`. PGJ uses its own local Hugging Face backend
configured in `configs/attacks/pgj.yaml`.

## Running

Fast pipeline check without learned models:

```bash
python main.py \
  --model mock \
  --attack identity \
  --defense none \
  --prompt "a blue rabbit mascot standing in a garden" \
  --target "blue rabbit mascot"
```

Groot against SAFREE on SDXL:

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

Batch run:

```bash
python main.py \
  --model diffusers \
  --model-config configs/models/sdxl.yaml \
  --attack groot \
  --defense safree \
  --prompt-file data/datasets/safety_nonsexual/safety_nonsexual_10.csv \
  --max-candidates 3
```

Real Diffusers runs automatically load `configs/evaluation/llm_judge.yaml`.
Use `policy_violation` for the non-sexual safety dataset. Use
`target_presence` only for benign target-preservation experiments.

## Benchmark Matrices

The final SDXL matrix contains five attack rows (`identity` plus four attacks)
and three defense columns (`none`, `safree`, and `latent_guard_lite`):

```bash
bash scripts/run_sdxl_safety_15.sh
```

On a four-GPU server:

```bash
bash scripts/run_sdxl_safety_15_4gpu.sh
```

TraSCE is evaluated separately with Stable Diffusion 1.4 because it is tied to
that pipeline. PGJ and Ring-A-Bell comparison scripts are also available under
`scripts/`.

Start with `safety_nonsexual_10.csv` before running the full 100-row dataset.

## Results

Each run writes:

- `results.jsonl`: compact result, one line per candidate
- `results.csv`: the same compact data for spreadsheets
- `details.jsonl`: full technical trace for debugging
- `config.yaml`: merged run configuration
- `images/`: images released by the defense pipeline

`success` is decided by the shared LLM evaluator. Defense behavior is recorded
separately through `prompt_blocked`, `image_blocked`, and `defense_bypassed`.

## Documentation

- [Components](docs/COMPONENTS.md): final attacks, defenses, and evaluator
- [Architecture](docs/ARCHITECTURE.md): pipeline and file responsibilities
- [Paper map](docs/PAPER_IMPLEMENTATION_MAP.md): ownership, sources, deviations
- [Groot](docs/GROOT.md), [DACA](docs/DACA.md), [Ring-A-Bell](docs/ring_a_bell.md)
- [SAFREE](docs/SAFREE.md), [TraSCE](docs/trasce.md)
- [Safety scope](docs/SAFETY_SCOPE.md)
- [Add an attack](docs/HOW_TO_ADD_ATTACK.md), [add a defense](docs/HOW_TO_ADD_DEFENSE.md)
