# Latent Guard: Paper and Implementation Comparison

Reference: [Latent Guard: a Safety Framework for Text-to-Image Models](https://arxiv.org/abs/2404.08031),
ECCV 2024. [Official implementation](https://github.com/rt219/LatentGuard).

The framework component is named `latent_guard_lite`. It adapts the released
inference architecture and pretrained weights; the name does not mean that
the trained attention mapping has been replaced with plain CLIP similarity.
It is not a reproduction of the training pipeline.

## What Is the Same

| Component | What is retained |
| --- | --- |
| Defense principle | Detect restricted semantic concepts in text before image generation, rather than checking only literal keywords. |
| Text encoder | Frozen `openai/clip-vit-large-patch14` representations. |
| Learned mapping | Concept queries attend to prompt keys and values; the concept and concept-conditioned prompt representations are projected into a shared space. |
| Mapping dimensions | 768-dimensional inputs, 16 heads with head dimension 32, and 128-dimensional outputs, matching the released checkpoint architecture. |
| Similarity | Normalized embeddings are compared with cosine similarity and the checkpoint's learned scale. The score is not a probability. |
| Decision structure | Score the prompt against the restricted concepts and block when the maximum reaches the configured threshold. |
| Model parameters | Released mapping checkpoint, without project retraining or fine-tuning. |
| Separate CoPro check | Official CoPro ID/OOD concept splits and explicit, synonym, and adversarial prompt evaluation are supported separately from image-generation experiments. The completed full evaluation closely matched the six reported AUCs; this supports inference fidelity, not training reproduction. |

## What Is Different

| Difference | Reason / implementation choice | Consequence for comparison |
| --- | --- | --- |
| Framework adapter instead of the original execution environment | Inference is integrated into `check_prompt`, with batched scoring, cached concept embeddings, and framework logging. | Execution and diagnostics differ; these changes are intended to preserve the scoring function. Parity testing checks that intention rather than assuming it. |
| Training is not reproduced | Load the released `model_parameters.pth` instead of rebuilding the training procedure. | Can claim pretrained inference reproduction, not independently reproduced training or checkpoint learning. |
| Project restricted-concept list | The final YAML uses `restricted_concepts_safety_nonsexual_100.yaml` for every prompt. `use_target_concept` and `include_aliases` are false. | The final attack-defense benchmark does not use the CoPro concept lists. Coverage of this fixed list affects blocking and generalization. The filename refers to the 100-prompt dataset, not necessarily 100 distinct concepts. |
| Fixed project operating point | The final preset uses threshold 9.0131 across the project benchmark without project-specific retraining. | Accuracy and blocking depend on this threshold and concept list. AUC agreement alone does not validate the operating point on a different dataset. |
| Different downstream benchmark | The final Latent Guard comparison uses SDXL, project attacks, the custom 100-prompt dataset, and the shared image judge. | Downstream attack success is a different measurement from CoPro text-classification AUC; their numbers are not interchangeable. |
| Fail-closed operational handling | With `fail_on_error: true`, loading/scoring failures prevent normal generation. | An operational failure is not evidence that the learned detector recognized a concept. Failed runs must not be reported as valid defense effectiveness. |

## Scope of the Claim

The strongest supported claim is **released-weight inference reproduction,
plus a separate project attack-defense evaluation**. Agreement on CoPro does
not establish identical behavior on every prompt or faithful reproduction of
every experiment in the paper. The safety-focused project dataset also does
not establish benign-prompt utility or a benign false-positive rate.

Implementation: `t2i_framework/defenses/latent_guard_lite.py`.
Final preset: `configs/defenses/latent_guard_lite.yaml`.
Parity check: `scripts/check_latent_guard_parity.py`.
CoPro evaluation: `scripts/evaluate_latent_guard_copro.py`.
Checkpoint provenance and setup remain in [the data README](../data/latent_guard/README.md).
