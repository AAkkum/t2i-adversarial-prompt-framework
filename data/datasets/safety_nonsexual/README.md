# Non-Sexual Safety Benchmark

`safety_nonsexual_100.csv` is the main safety benchmark for the final
attack/defense experiments. It contains 100 prompts, 20 target concepts and five
different scenes per target.

## Composition

| Category | Prompts | Targets |
| --- | ---: | ---: |
| Violence | 20 | 4 |
| Illegal activity | 20 | 4 |
| Self-harm | 15 | 3 |
| Shocking content | 15 | 3 |
| Hate | 15 | 3 |
| Harassment | 15 | 3 |

Sexual content is deliberately excluded. Every prompt is an image description,
not an instruction for committing harm. Every prompt also contains its exact
`target_concept`, so all attacks start from the same explicit baseline.

The prompts are project-authored. The category taxonomy was selected after an
audit of the I2P benchmark from *Safe Latent Diffusion*. The I2P prompts were not
copied because many rows are noisy, ambiguous or unsuitable for target-specific
attack testing. This is therefore a controlled project benchmark, not an I2P
reproduction or an official paper benchmark.

Audited I2P revision:

- Source: `AIML-TUDA/i2p`, file `i2p_benchmark.csv`
- Revision: `75c00f8a73fdbccc21981d4400d9061d4315a89c`
- SHA-256: `5e78f211b3944d90182ab708ed0bd8af657c24a308153e6ccd5db9d9dbfe50cb`

## Files

- `safety_nonsexual_10.csv`: smoke test covering all six categories.
- `safety_nonsexual_25.csv`: one row for every target plus five extra variants.
- `safety_nonsexual_100.csv`: final experiment batch.

The subsets are nested: every 10-row case is in the 25-row file and every
25-row case is in the 100-row file.

CSV columns:

- `id`: stable case identifier.
- `prompt`: input to the attack.
- `target_concept`: concept the attack tries to preserve while bypassing a defense.
- `category`: safety category used for reporting.
- `source`: prompt provenance; every current row is `project_authored`.

## Component Support

- Groot, PGJ, DACA and TextFooler receive `prompt` and `target_concept` directly.
- Search Attack descriptions are in `data/search_attack/concept_targets.json`.
- Ring-A-Bell pairs are in
  `data/ring_a_bell/concept_pairs_safety_nonsexual_100.json`; use
  `configs/attacks/ring_a_bell.yaml`.
- CharacterFilter, SAFREE and TraSCE use the row's `target_concept` directly.
- LatentGuard uses `configs/defenses/latent_guard_safety_nonsexual.yaml`.

The shared evaluator in `configs/evaluation/llm_judge.yaml` uses
`policy_violation` for this dataset. `target_presence` only asks whether the
target appears and does not by itself establish that an unsafe attack succeeded.

Start with the 10-row subset before expensive 100-row image runs:

```bash
python main.py \
  --model diffusers \
  --model-config configs/models/sdxl.yaml \
  --attack groot \
  --attack-config configs/attacks/groot.yaml \
  --defense safree \
  --defense-config configs/defenses/safree.yaml \
  --prompt-file data/datasets/safety_nonsexual/safety_nonsexual_10.csv
```

## Limits

- The labels and target phrases were reviewed manually, not by independent annotators.
- The dataset measures these 20 selected concepts, not all possible unsafe content.
- Some content is disturbing even though sexual content is absent.
- The existing `representative_prompt_batch_100.csv` remains useful as a benign
  target-preservation/utility benchmark; do not combine its results with this
  safety benchmark's policy-violation rate.
