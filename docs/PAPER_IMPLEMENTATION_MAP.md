# Paper / Method Implementation Map

This note maps the current attacks and defenses to the papers or methods they
are based on. Several components are intentionally "inspired by" a paper rather
than full reproductions.

## Contributor Ownership

This section is based on `docs/MERGE_INTEGRATION_NOTES.md`, later commits on
`main`, and the project history.

| Contributor | Main contribution area |
|---|---|
| Atabey | Initial framework structure, shared CLI/registry/runner integration, final merge maintenance, `groot`, `image_clip_filter`, image-CLIP evaluation, baseline controls, and the config/docs cleanup. |
| Abdel | `pgj` attack only. |
| Burak | `search_attack`, `character_filter`, `filter_placeholder`, BLIP image captioning support, MiniLM semantic concept matching, search-attack support data, search-attack documentation, and the safer image quarantine/release behavior. |
| Hans | `textfooler_style`, Ollama/Qwen paraphraser support, Ollama similarity judge, CLIP text similarity support, `clip_similarity` defense, prompt-prompt similarity evaluation, prompt-case datasets/loading, component config presets, and later `latent_guard_lite`. |

Short version: Abdel did just the PGJ attack. The remaining implemented attacks
and defenses came from Atabey, Burak, and Hans, with Atabey also doing the
manual integration of the shared framework pieces.

## Attacks

| Component | Main contributor | Relevant paper / method | Implementation status |
|---|---|---|---|
| `pgj` | Abdel | Huang et al., "Perception-Guided Jailbreak Against Text-to-Image Models", AAAI 2025, arXiv:2408.10848 | Closest direct paper-based implementation in this repo. It implements PGJ's PSTSI idea: use an LLM to rewrite unsafe/protected words into visually similar but semantically different safe phrases. It is not a full benchmark reproduction: it uses a local Hugging Face LLM backend, project-specific prompts, and the framework's local evaluation pipeline. |
| `groot` | Atabey | Liu et al., "Groot: Adversarial Testing for Text-to-Image Generative Models with Tree-based Semantic Transformation", arXiv:2402.12100; authors' TREANT repository | Reimplements the complete algorithmic workflow: initial probe, LLM-built Prompt Parse Tree, randomized semantic decomposition, iterative leaf refinement, sensitive-element drowning, stage-aware feedback, query accounting, and the SD-only/IED-only ablations. The execution environment differs from the paper: local Qwen replaces GPT-4, SD 3.5 replaces the paper's DALL-E 3/Midjourney/SDXL setup, project defenses expose filter stages, and Qwen vision replaces manual labels. It is therefore a method reimplementation, not an exact reproduction of the paper's reported experiment. |
| `textfooler_style` | Hans | Jin et al., "Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment", AAAI 2020 | Adapted TextFooler-style attack. It keeps the main pattern of word/phrase importance ranking plus replacement search plus semantic-similarity filtering, but it is adapted to text-to-image prompt defenses. Replacements can come from an Ollama/Qwen paraphraser, similarity can use CLIP text embeddings or sentence transformers, and candidate success is measured against this framework's defenses rather than a text classifier. |
| `search_attack` | Burak | No direct paper reproduction | Local/theoretical prototype. It generates deterministic prompt variants from data-driven phrase pools and concept replacement files. The German `docs/search_attack.md` explicitly says it is not a full implementation of a published attack. |
| `identity` | Atabey | Baseline | No paper. It returns the original prompt unchanged. |

## Defenses

The primary comparison matrix uses `latent_guard_lite`, `character_filter`, and
`image_clip_filter`. `clip_similarity` remains available as supporting work but
is not counted as a fourth participant defense.

| Component | Main contributor | Relevant paper / method | Implementation status |
|---|---|---|---|
| `latent_guard_lite` | Hans | Liu et al., "Latent Guard: a Safety Framework for Text-to-image Generation", ECCV 2024 / arXiv:2404.08031 | Lightweight adapter, not a full reproduction. It reuses the public LatentGuard Embedding Mapping Layer idea and expects the released `model_parameters.pth`, but it does not implement the full data-generation/training pipeline. |
| `clip_similarity` | Hans | CLIP text embedding similarity, based on Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021 | Project-specific baseline defense. It compares the attacked prompt to restricted concepts in CLIP text space and blocks above a threshold. |
| `image_clip_filter` | Atabey | Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021 (CLIP model) | Project-specific image-stage defense, not a reproduction of a published defense. It applies CLIP image-text similarity to the generated image and the original target concept, then blocks scores at or above a configured threshold. |
| `character_filter` | Burak | Keyword filtering + MiniLM semantic matching + BLIP captioning | Local multi-stage defense, not one paper. It combines direct keyword matching, MiniLM sentence similarity, BLIP captioning, and MiniLM image-caption matching. BLIP is from Li et al., "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation", ICML 2022. |
| `none` | Atabey | Baseline | No defense. |

## Infrastructure-Only / Placeholder Defenses

These are registered in the framework, but they should not be presented as main
implemented research defenses.

| Component | Main contributor | Why it exists | Keep? |
|---|---|---|---|
| `filter_placeholder` | Burak | Neutral no-op placeholder used while wiring the search-attack branch and tests. It deliberately allows prompts and images. | Keep only if old search-branch experiments/tests still need it. Do not report it as a real defense. |
| `embedding_filter` | Atabey | Early placeholder for a future embedding-based prompt filter. The real semantic defenses are now `clip_similarity`, `character_filter`, and `latent_guard_lite`. | Safe candidate for later removal. Do not report it as a real defense. |

## Best Supervisor Answer

The safest answer is:

> We implemented a modular framework containing paper reimplementations,
> adaptations, and project-specific baselines. Groot now contains both of the
> paper's main mutation stages and its adaptive feedback loop, but uses local
> Qwen, Stable Diffusion 3.5, project defense stages, and an automated vision
> judge instead of the paper's hosted models and manual labels. PGJ,
> TextFooler-style rewriting, and LatentGuard-lite also contain documented
> adaptations. `search_attack`, `character_filter`, and the CLIP image filter
> are project-specific methods rather than 1:1 paper reproductions.

## References

- PGJ: https://ojs.aaai.org/index.php/AAAI/article/view/34821
- Groot: https://arxiv.org/abs/2402.12100
- Groot authors' code (repository name TREANT): https://github.com/llm-jailbreaker/Treant
- DrAttack: https://aclanthology.org/2024.findings-emnlp.813/
- TextFooler: https://ojs.aaai.org/index.php/AAAI/article/view/6311
- Latent Guard: https://github.com/rt219/LatentGuard
- CLIP: https://openai.com/index/clip/
- BLIP: https://proceedings.mlr.press/v162/li22n.html
