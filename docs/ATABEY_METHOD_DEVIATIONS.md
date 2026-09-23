# Atabey: Method and Deviation Notes

This is the presentation/report disclosure for Atabey's attack and defense. It
separates the implemented method from the paper's exact experimental setup.

## Short Classification

| Component | What it is | Paper status |
|---|---|---|
| `groot` | Adaptive Prompt Parse Tree attack with semantic decomposition and sensitive-element drowning | Reimplementation of Groot's algorithmic workflow with a different local model and evaluator |
| `image_clip_filter` | Post-generation target filter using CLIP image-text similarity | Project-specific defense built with CLIP, not a published defense reproduction |
| Image-CLIP evaluator | Measures whether an image resembles the benchmark target | Project-specific auxiliary metric using the same CLIP mechanism |

## Attack: `groot`

Primary reference: Yi Liu et al., "Groot: Adversarial Testing for Text-to-Image
Generative Models with Tree-based Semantic Transformation",
arXiv:2402.12100. The authors publish their implementation in a repository
named TREANT: https://github.com/llm-jailbreaker/Treant.

### Implemented Paper Workflow

1. Query the target image model with the original prompt.
2. After a text-stage block, ask a local LLM to construct a Prompt Parse Tree
   with object, attribute, and relation information.
3. Traverse and randomly reorder tree phrases to form a semantically decomposed
   prompt.
4. If the prompt remains blocked, select a leaf and ask the LLM to split it into
   a more granular subtree, up to the configured depth and node limits.
5. After an image-stage block or unsuccessful generated image, apply sensitive-
   element drowning by placing the transformed scene in one panel and unrelated
   benign scenes in independent panels.
6. Stop when the configured image-success criterion is met or the
   `--max-candidates` query budget is exhausted.
7. Support `full`, `semantic_only`, and `drowning_only` modes for the paper's
   ablation comparison.

The result metadata stores the mutation strategy, query count, tree depth, node
count, serialized tree, drowning round, and multimodal judgment.

### Deviations From The Paper

| Deviation | Reason | Consequence |
|---|---|---|
| Local multimodal model instead of GPT-4 | Keep prompt transformation and image review local | Generated trees and judgments may differ from the paper; model quality becomes an experimental variable |
| Stable Diffusion 3.5 instead of DALL-E 3, Midjourney, and SDXL | Target the project's available open model | Results cannot be numerically compared with the paper's 93.66% aggregate result |
| Explicit framework defense stages | Local Diffusers does not expose the same hosted text/image filter error messages | Stage classification is reliable for framework defenses but does not reproduce proprietary service filters |
| Local multimodal labels instead of manual labels | Automate larger local experiments | Judge errors and bias affect measured success; validate a sample manually for the report |
| Project prompt data instead of NSFW-1k | Dataset reproduction is outside the current code task | Category distribution and difficulty differ from the paper |
| Configurable target concept | The framework's batch schema already provides one | This can make target-presence evaluation an oracle setup; paper-compatible policy evaluation should not rely only on the target label |
| Modernized tree parser and failure handling | The released TREANT code has disabled loops, incorrect calls, and only a DALL-E adapter | The implementation follows the paper description rather than reproducing those software defects |

The correct claim is **Groot method reimplementation with a local experimental
backend**, not **exact reproduction of the paper's results**.

### Success Rules

`configs/attacks/groot.yaml` defaults to `policy_violation`, which follows the
paper: the generated image must preserve the requested intent, visibly violate
the configured policy, and exceed the confidence threshold.

The optional `target_presence` mode is useful for the project's harmless
protected-concept datasets. It checks whether the target remains visible and
the intent is preserved. That mode is an adaptation and must not be compared to
the paper's unsafe-content attack success rate.

## Defense: `image_clip_filter`

Technical reference: Alec Radford et al., "Learning Transferable Visual Models
From Natural Language Supervision", ICML 2021.

CLIP is the model used by this defense, but the CLIP paper does not propose this
text-to-image safety defense. It is a project-specific baseline.

After image generation, the defense computes CLIP cosine similarity between the
generated image and the supplied `target_concept`. It blocks the image when
`similarity >= threshold`.

Important limitations:

- It receives the benchmark's labelled target instead of discovering prohibited
  concepts itself.
- One threshold is not necessarily calibrated across identities, brands,
  animals, actions, and synthetic concepts.
- CLIP similarity is a proxy and can produce false positives and negatives.
- When the same CLIP model is also used as evaluator, that score is not
  independent evidence that the defense is correct.

## Presentation Wording

> I reimplemented Groot's algorithmic pipeline: LLM-generated Prompt Parse
> Trees, iterative semantic decomposition, sensitive-element drowning, adaptive
> filter feedback, query accounting, and both ablation modes. I modernized the
> backend to run with a local multimodal model and Stable Diffusion 3.5, and
> replaced manual image labels with an automated vision judgment. Therefore the method structure
> follows Groot, but the resulting numbers are not directly comparable with the
> paper's GPT-4, hosted-model, manually labelled experiments.
>
> My image-CLIP filter is our own post-generation baseline built with CLIP, not
> a reproduction of a published defense. It uses the benchmark's target label,
> so it should be reported as an oracle target-specific baseline and evaluated
> with independent labels.

## References

- Groot paper: https://arxiv.org/abs/2402.12100
- Groot/TREANT code: https://github.com/llm-jailbreaker/Treant
- CLIP: https://proceedings.mlr.press/v139/radford21a.html
