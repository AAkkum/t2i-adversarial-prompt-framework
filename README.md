# T2I Adversarial Prompt Framework

University project for testing prompt attacks and defenses against local
text-to-image models.

## Components

Main attacks:

- `daca`
- `groot`
- `pgj`
- `search_attack`
- `textfooler_style`

Main defenses:

- `character_filter`
- `latent_guard_lite`

`identity` and `none` are controls. One shared local multimodal LLM evaluates
the generated images. CLIP is not used as the experiment evaluator.

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
`http://127.0.0.1:8082/v1` with API alias `local-llm`.

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

Real Diffusers runs automatically load `configs/evaluation/llm_judge.yaml`.
Its default `target_presence` rule asks whether the target is visible and the
original scene is preserved. Change it to `policy_violation` only for a
policy-labelled benchmark intended to follow Groot's paper evaluation.

Batch run:

```bash
python main.py \
  --model diffusers \
  --model-config configs/models/sdxl.yaml \
  --attack groot \
  --defense character_filter \
  --prompt-file data/datasets/representative/representative_prompt_batch_100.csv \
  --max-candidates 3
```

Start with `data/tmp_synthetic_10.csv` before an expensive 100-row run.

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
- `docs/search_attack.md`: Search Attack behavior
- `docs/PAPER_IMPLEMENTATION_MAP.md`: owners, papers, and deviations
- `docs/DEFENSE_PAPER_SELECTION.md`: candidate paper for the next defense
- `docs/HOW_TO_ADD_DEFENSE.md`: minimal defense implementation steps
- `docs/SAFETY_SCOPE.md`: project safety boundary
