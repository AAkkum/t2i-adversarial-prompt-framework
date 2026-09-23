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
