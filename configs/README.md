# Configuration

Configuration is separated by responsibility:

- `models/`: Diffusers model, scheduler, image size, and memory settings
- `attacks/`: attack-specific settings
- `defenses/`: defense-specific settings
- `evaluation/llm_judge.yaml`: shared image-evaluation rule
- `local_llm.yaml`: shared multimodal server settings

The CLI automatically loads a YAML file whose name matches the selected
component. Explicit files and `--config` are merged afterward, so later values
override earlier values.

```bash
python main.py \
  --model diffusers \
  --model-config configs/models/sdxl.yaml \
  --attack groot \
  --defense safree \
  --prompt-file data/datasets/safety_nonsexual/safety_nonsexual_10.csv
```

The final dataset uses:

- `configs/attacks/ring_a_bell.yaml` for its matching concept-pair file
- `configs/defenses/latent_guard_lite.yaml` for its matching concept list
- `configs/evaluation/llm_judge.yaml` with `policy_violation`

`scripts/start-local-llm.sh` reads `configs/local_llm.yaml`. DACA has a separate
`daca_llm` section in `configs/attacks/daca.yaml` and a separate start script.
