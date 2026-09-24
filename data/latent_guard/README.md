# LatentGuard Weights

This folder is reserved for optional LatentGuard pretrained weights.

The `latent_guard_lite` defense expects the released LatentGuard state dict at:

```text
data/latent_guard/model_parameters.pth
```

The official checkpoint used for local parity validation has:

```text
SHA256 02315494F35E819DC627BA858408D571E06A594590E8BD4026BA63F0922DA231
```

Download it from the official LatentGuard repository:

```text
https://github.com/rt219/LatentGuard
```

The framework does not commit the `.pth` file because it is an external model
artifact.

Set `fail_on_error: true` for evaluation runs so missing weights, model-loading
failures, and incompatible checkpoints stop the experiment instead of allowing
prompts. Concept embeddings are cached after their first use and reused for the
remaining prompts in the process.

The default defense config uses the fixed representative-batch blacklist at:

```text
data/latent_guard/restricted_concepts_representative_100.yaml
```

It contains all 100 unique targets from
`data/datasets/representative/representative_prompt_batch_100.csv`. With
`use_target_concept: false`, every prompt is checked against this same list.

## Official parity check

Run the independent official-reference calculation and compare it with the
framework scorer:

```bash
python scripts/check_latent_guard_parity.py
```

The command compares each score within configured floating-point tolerances and
also requires the final threshold decision to match. It defaults to eight
concepts to keep the check quick; pass `--max-concepts 0` to compare all 100.

## Official CoPro evaluation

Download the official `CoPro_v1.0.json` into this directory, then extract the
paper's 578 ID and 145 OOD concepts:

```bash
python scripts/prepare_latent_guard_copro.py
```

The generated `restricted_concepts_copro_id.yaml` and
`restricted_concepts_copro_ood.yaml` files can be selected through the matching
configs in `configs/defenses/`. They preserve the paper protocol by disabling
aliases and runtime target concepts.

Run a small end-to-end smoke evaluation before starting the complete protocol:

```bash
python scripts/evaluate_latent_guard_copro.py --limit 10 --output results/latent_guard_copro/smoke
```

Run all ID/OOD Explicit, Synonym, and Adversarial conditions with:

```bash
python scripts/evaluate_latent_guard_copro.py --device cuda:0 --output results/latent_guard_copro/full
```

The evaluator writes resumable scores, per-example predictions, and a summary
containing AUC, threshold accuracy, unsafe recall, safe recall, and the AUC
difference from Table 1b. This is a prompt-classification evaluation and does
not load an image generator or local LLM. Re-run the same command with the same
output directory to continue from the latest score checkpoint.
