# Latent Guard Data

`latent_guard_lite` expects the released state dictionary at:

```text
data/latent_guard/model_parameters.pth
```

The checkpoint is external and is therefore not committed. The checkpoint used
for local parity validation has SHA-256:

```text
02315494F35E819DC627BA858408D571E06A594590E8BD4026BA63F0922DA231
```

Source: https://github.com/rt219/LatentGuard

The final project config, `configs/defenses/latent_guard_lite.yaml`, uses
`restricted_concepts_safety_nonsexual_100.yaml`. It checks every prompt against
the same fixed list and fails closed when weights or model loading fail.

## Parity Check

Compare the framework scorer with an independent official-reference
calculation:

```bash
python scripts/check_latent_guard_parity.py
```

The default check uses eight concepts. Pass `--max-concepts 0` to check the
complete final concept list.

## Official CoPro Evaluation

For a separate paper-protocol check, place the official `CoPro_v1.0.json` in
this directory and prepare its ID/OOD concept lists:

```bash
python scripts/prepare_latent_guard_copro.py
python scripts/evaluate_latent_guard_copro.py --limit 10 \
  --output results/latent_guard_copro/smoke
```

This CoPro evaluation measures prompt classification only. It does not load an
image generator or the shared multimodal evaluator.
