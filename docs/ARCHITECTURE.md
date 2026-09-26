# Architecture

The pipeline has five jobs:

```text
prompt
  -> attack creates candidate prompt
  -> defense checks prompt
  -> image model generates image
  -> defense checks image
  -> local multimodal LLM evaluates result
```

The defense and evaluator answer different questions:

- A defense says `allowed` or `blocked`.
- The evaluator says whether the attack objective was achieved.

The evaluator returns `target_visible`, `intent_preserved`,
`policy_violation`, `confidence`, and a short reason. The configured success
rule converts those values into the final `success` field.

## Main Files

- `main.py`: command-line options and config loading
- `t2i_framework/core/registry.py`: available components
- `t2i_framework/evaluation/runner.py`: executes the pipeline
- `t2i_framework/evaluation/llm_image_judge.py`: shared evaluator
- `t2i_framework/evaluation/result_writer.py`: compact and detailed output
- `t2i_framework/attacks/`: attack implementations plus the identity control
- `t2i_framework/defenses/`: defense implementations plus the none control
- `t2i_framework/models/`: mock and Diffusers adapters

Groot is adaptive. After each unsuccessful candidate, the runner gives it the
result and Groot may create another prompt. Other attacks generate their
candidates before evaluation.

TraSCE and SAFREE are generation-time defenses. Their prompt checks prepare a
request, and the Diffusers adapter calls their generation hook while denoising.

## Configuration

- `--model-config`: image model settings
- `--attack-config`: attack settings
- `--defense-config`: defense settings
- `--config`: optional evaluation overrides

Real Diffusers runs load `configs/evaluation/llm_judge.yaml` by default. Mock
runs skip the learned evaluator. Both Groot and the evaluator connect using
`configs/local_llm.yaml`.
