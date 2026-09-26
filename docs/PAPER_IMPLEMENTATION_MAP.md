# Paper And Ownership Map

## Attacks

| Component | Owner | Basis | Important deviation |
|---|---|---|---|
| `daca` | Hans | DACA, arXiv:2312.07130v4 | Exact released helper-prompt assets; selectable paper-algorithm and official-release orchestration; shared local LLM and local target models remain experimental substitutions |
| `groot` | Atabey | Groot/TREANT, arXiv:2402.12100 | Local LLM, local Diffusers models, automated evaluator, project datasets |
| `pgj` | Abdel | PGJ, AAAI 2026 | Local Hugging Face backend and framework-specific prompts |
| `search_attack` | Burak | Project-specific method | No direct paper reproduction |
| `textfooler_style` | Hans | TextFooler, AAAI 2020 | Adapted from text classification to text-to-image prompts |

## Defenses

| Component | Owner | Basis | Important deviation |
|---|---|---|---|
| `character_filter` | Burak | Project-specific keyword, MiniLM, and BLIP filter | No direct paper reproduction |
| `latent_guard_lite` | Hans | Latent Guard, ECCV 2024 | Adapter around released architecture/weights, not full training reproduction |
| `safree` | Atabey | SAFREE, ICLR 2025 | Full three-stage SDXL port. Projection follows released SDXL behavior; self-validation and latent re-attention are ported from the paper and released SD 1.4 path because they are not wired into the released SDXL path. |

## Shared Evaluator

Atabey implemented the local multimodal-LLM evaluator used after image
generation. It replaces Groot's manual image labelling and is applied to every
attack so that success has one definition within an experiment.

## References

- DACA: https://arxiv.org/abs/2312.07130
- DACA code: https://github.com/researchcode001/daca
- Groot: https://arxiv.org/abs/2402.12100
- TREANT code: https://github.com/llm-jailbreaker/Treant
- PGJ: https://ojs.aaai.org/index.php/AAAI/article/view/34821
- TextFooler: https://ojs.aaai.org/index.php/AAAI/article/view/6311
- Latent Guard: https://github.com/rt219/LatentGuard
- SAFREE: https://openreview.net/forum?id=hgTFotBRKl
- SAFREE code: https://github.com/jaehong31/SAFREE
- BLIP: https://proceedings.mlr.press/v162/li22n.html
