# Configuration

Configuration is separated by responsibility:

- `configs/models/`: Diffusers model and memory settings
- `configs/attacks/`: attack settings
- `configs/defenses/`: defense settings
- `configs/evaluation/`: shared evaluator rules
- `configs/local_llm.yaml`: local model and server settings

The CLI automatically loads the matching model, attack, and defense file when
one exists. Real Diffusers runs also load `configs/evaluation/llm_judge.yaml`.
An explicit `--config` is merged on top.

Examples:

```bash
python main.py --model diffusers --model-config configs/models/sdxl.yaml ...
python main.py --attack groot --attack-config configs/attacks/groot.yaml ...
python main.py --defense character_filter \
  --defense-config configs/defenses/character_filter.yaml ...
```

`configs/evaluation/llm_judge.yaml` defines how generated images are judged.
It does not select or start the local model.

`scripts/start-local-llm.sh` reads the server launch settings from
`configs/local_llm.yaml`. A Hugging Face model spec is downloaded and cached by
llama.cpp on first use, so there is no separate download script.
