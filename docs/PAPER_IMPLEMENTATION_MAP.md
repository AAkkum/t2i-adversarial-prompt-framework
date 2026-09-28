# Paper And Ownership Map

## Attacks

| Component | Owner | Basis | Main deviation |
|---|---|---|---|
| `daca` | Hans | DACA, arXiv:2312.07130 | Released prompts and selectable orchestration are retained, but local LLMs, local image models, and project evaluation are substituted |
| `groot` | Atabey | Groot/TREANT, arXiv:2402.12100 | Local LLM and Diffusers models, automated evaluation, and project datasets replace hosted models and manual labels |
| `pgj` | Abdel | PGJ, AAAI 2026 | Uses a local Hugging Face backend and framework-specific prompts |
| `ring_a_bell` | Burak | Ring-A-Bell, ICLR 2024 | Uses project-created concept pairs and the framework's SD 1.4/evaluation pipeline |
| `identity` | Shared control | No attack | Sends the prompt unchanged |

## Defenses

| Component | Owner | Basis | Main deviation |
|---|---|---|---|
| `latent_guard_lite` | Hans | Latent Guard, ECCV 2024 | Adapter around released architecture and weights, not reproduction of the training pipeline |
| `safree` | Atabey | SAFREE, ICLR 2025 | Full three-stage SDXL port; two stages are ported from the released SD 1.4 path because the released SDXL path does not wire them in |
| `trasce` | Burak | TraSCE, arXiv:2412.07658 | Diffusers integration follows the released executable sampler on SD 1.4 and uses project prompts/evaluation |
| `none` | Shared control | No defense | Allows the prompt and generated image |

## Shared Evaluator

Atabey implemented the local multimodal-LLM evaluator used after image
generation. It replaces manual image labelling and gives all components the same
success definition within each experiment. This is project evaluation code, not
part of the attack or defense papers.

## References

- DACA: https://arxiv.org/abs/2312.07130
- DACA code: https://github.com/researchcode001/daca
- Groot: https://arxiv.org/abs/2402.12100
- TREANT code: https://github.com/llm-jailbreaker/Treant
- PGJ: https://ojs.aaai.org/index.php/AAAI/article/view/34821
- Ring-A-Bell: https://openreview.net/forum?id=lm7MRcsFiS
- Ring-A-Bell code: https://github.com/chiayi-hsu/Ring-A-Bell
- Latent Guard: https://arxiv.org/abs/2404.08031
- Latent Guard code: https://github.com/rt219/LatentGuard
- SAFREE: https://openreview.net/forum?id=hgTFotBRKl
- SAFREE code: https://github.com/jaehong31/SAFREE
- TraSCE: https://arxiv.org/abs/2412.07658
- TraSCE code: https://github.com/SonyResearch/TraSCE
