# Data

The final experiments use one project-authored, non-sexual safety dataset:

- `datasets/safety_nonsexual/safety_nonsexual_10.csv`: quick smoke batch
- `datasets/safety_nonsexual/safety_nonsexual_25.csv`: intermediate batch
- `datasets/safety_nonsexual/safety_nonsexual_100.csv`: final benchmark

Each CSV row contains `id`, `prompt`, `target_concept`, `category`, and
`source`. See the dataset README for its composition, provenance, and limits.

## Component Data

- `daca/reference/`: helper prompts copied from the official DACA release
- `latent_guard/model_parameters.pth`: external checkpoint location; not committed
- `latent_guard/restricted_concepts_safety_nonsexual_100.yaml`: final concept list
- `latent_guard/restricted_concepts_copro_*.yaml`: official parity datasets
- `ring_a_bell/concept_pairs_safety_nonsexual_100.json`: final concept pairs

Groot creates its prompt tree at runtime and needs no decomposition file.

## Smoke Run

```bash
python main.py \
  --model mock \
  --attack identity \
  --defense none \
  --prompt-file data/datasets/safety_nonsexual/safety_nonsexual_10.csv
```
