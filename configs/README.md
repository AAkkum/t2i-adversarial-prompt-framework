# Configs

Configs are intentionally separated by responsibility.

## Model Configs

Use with `--model-config`.

- `configs/models/mock.yaml`
- `configs/models/sdxl.yaml`
- `configs/models/sd35_medium.yaml`
- `configs/models/sd35_large.yaml`
- `configs/models/flux.yaml`

Example:

```bash
python main.py --model diffusers --model-config configs/models/sdxl.yaml ...
```

## Attack Configs

Use with `--attack-config` when you want a non-default attack config.

The CLI also auto-loads `configs/attacks/<attack>.yaml` when it exists.

- `configs/attacks/groot_lite.yaml`
- `configs/attacks/groot_lite_external.yaml`
- `configs/attacks/search_attack.yaml`
- `configs/attacks/textfooler_style.yaml`
- `configs/attacks/pgj.yaml`

Example:

```bash
python main.py --attack groot_lite --attack-config configs/attacks/groot_lite_external.yaml ...
```

## Defense Configs

Use with `--defense-config` when you want a non-default defense config.

The CLI also auto-loads `configs/defenses/<defense>.yaml` when it exists.

- `configs/defenses/clip_similarity.yaml`
- `configs/defenses/clip_similarity_black_box.yaml`
- `configs/defenses/character_filter.yaml`
- `configs/defenses/normalize_keywords.yaml`

## Evaluation Configs

Use with `--config`.

Evaluation configs should not contain `model`, `attack`, or `defense` sections.

- `configs/evaluation/image_clip.yaml`
- `configs/evaluation/prompt_similarity.yaml`
- `configs/evaluation/clip_and_prompt_similarity.yaml`

Example:

```bash
python main.py \
  --model diffusers \
  --model-config configs/models/sdxl.yaml \
  --attack groot_lite \
  --defense none \
  --config configs/evaluation/clip_and_prompt_similarity.yaml \
  --prompt "a blue rabbit mascot standing in a garden" \
  --target "blue rabbit mascot"
```

## Rule Of Thumb

- Model choice and size belongs in `configs/models/`.
- Attack parameters belong in `configs/attacks/`.
- Defense thresholds/settings belong in `configs/defenses/`.
- Metrics/evaluation settings belong in `configs/evaluation/`.
- Avoid top-level combined experiment configs unless there is a strong reason.
