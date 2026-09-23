# Paper And Ownership Map

## Attacks

| Component | Owner | Basis | Important deviation |
|---|---|---|---|
| `groot` | Atabey | Groot/TREANT, arXiv:2402.12100 | Local LLM, local Diffusers models, automated evaluator, project datasets |
| `pgj` | Abdel | PGJ, AAAI 2026 | Local Hugging Face backend and framework-specific prompts |
| `search_attack` | Burak | Project-specific method | No direct paper reproduction |
| `textfooler_style` | Hans | TextFooler, AAAI 2020 | Adapted from text classification to text-to-image prompts |

## Defenses

| Component | Owner | Basis | Important deviation |
|---|---|---|---|
| `character_filter` | Burak | Project-specific keyword, MiniLM, and BLIP filter | No direct paper reproduction |
| `latent_guard_lite` | Hans | Latent Guard, ECCV 2024 | Adapter around released architecture/weights, not full training reproduction |

## Shared Evaluator

Atabey implemented the local multimodal-LLM evaluator used after image
generation. It replaces Groot's manual image labelling and is applied to every
attack so that success has one definition within an experiment.

## References

- Groot: https://arxiv.org/abs/2402.12100
- TREANT code: https://github.com/llm-jailbreaker/Treant
- PGJ: https://ojs.aaai.org/index.php/AAAI/article/view/34821
- TextFooler: https://ojs.aaai.org/index.php/AAAI/article/view/6311
- Latent Guard: https://github.com/rt219/LatentGuard
- BLIP: https://proceedings.mlr.press/v162/li22n.html
