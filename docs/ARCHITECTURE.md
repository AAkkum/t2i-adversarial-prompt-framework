# Architecture

The pipeline is:

```text
original prompt
  -> Attack module
  -> Defense module, pre-generation
  -> Image model adapter
  -> Defense module, post-generation
  -> Evaluator / logger
  -> saved image + JSON/CSV result rows
```

The CLI selects one registered model, attack, and defense by name. `ExperimentRunner` coordinates the selected components and writes outputs through `ResultWriter`.

Core extension points:

- `t2i_framework/models/`: local Diffusers adapters, mock models, and API-backed models
- `t2i_framework/attacks/`: prompt rewriting or search strategies
- `t2i_framework/defenses/`: prompt filters, image filters, or composite defenses
- `t2i_framework/evaluation/`: result writing and metrics

The mock model is intentionally offline and CPU-only. It renders the selected prompt into a placeholder PNG so the framework can be tested without GPU or model downloads.
