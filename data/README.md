# Data

This folder contains prompt datasets and framework support data.

## What Hans Added

Hans added the larger prompt-case files used for batch evaluation. These files let you run many prompts in one command instead of launching the framework once per prompt.

The important idea is:

- each row is one prompt case;
- `prompt` is the original prompt sent into the attack;
- `target_concept` is the concept that should still appear after rewriting;
- `id`/`case_id` and `category` make results easier to group later;
- extra columns are kept as metadata in `results.jsonl`.

This is separate from attack support data. Prompt-case CSVs define experiment inputs. Files such as `attack_terms.yaml` and `data/search_attack/*.json` configure attacks or defenses. Groot generates decompositions at runtime and has no decomposition data file.

## Main Batch Files

- `all_prompt_cases.csv`
  Combined framework-ready prompt CSV. Use this when you want one large batch across the available datasets.

## Dataset Folders

- `datasets/synthetic/synthetic_prompt_batch_300.csv`
  Controlled synthetic prompts for debugging attacks, defenses, and evaluation metrics.

- `datasets/unbranding/unbranding_sample_10_per_brand.csv`
  Balanced 120-row sample from the public UNBRANDING dataset: 12 brands, 10 prompts per brand.

- `datasets/celebcaption/celebcaption_sample_5_per_person.csv`
  750-row sample from CelebCaption captions: 150 public figures, 5 prompts per person.

- `datasets/copyrighted_characters/copyrighted_characters_sample_5_per_character.csv`
  Manual 150-row prompt set for fictional character-name defense testing.

- `datasets/political_figures/political_figures_sample_5_per_person.csv`
  Manual 200-row prompt set for public-figure defense testing: 40 people, 5 prompts per person, balanced across 4 subcategories.

- `datasets/animals/animal_prompt_batch_300.csv`
  Manual 300-row prompt set for recognizable animal defense testing: 100 animals, 3 prompts per animal, covering land, water, flying, reptile/amphibian, and insect cases.

- `datasets/representative/representative_prompt_batch_100.csv`
  Compact 100-row benchmark batch with 100 unique targets, sampled across synthetic, animal, unbranding, copyrighted-character, political/public-figure, and celebcaption datasets.

## Support Files

- `restricted_concepts.yaml`
  Restricted concepts and aliases used by defenses.

- `attack_terms.yaml`
  Shared word lists used by TextFooler-style attacks and paraphrasers.

- `search_attack/`
  Burak's search-attack support data:
  - `blocked_terms.txt`: direct terms for character filtering
  - `concept_targets.json`: protected concept descriptions
  - `variant_phrases.json`: phrase fragments used to generate search variants
  - `test_cases.json`: small search-attack examples

- `latent_guard/`
  Optional local folder for the LatentGuard pretrained `model_parameters.pth` file used by `latent_guard_lite`.

- `example_prompts.csv`
  Tiny smoke-test CSV.

## Combined CSV Schema

`all_prompt_cases.csv` keeps the framework-required columns first:

```csv
id,prompt,target_concept,category,dataset
```

Extra metadata columns are kept when available:

```csv
difficulty,template_id,prompt_set,seed,source_filename,caption_variant,image_file,source_file,franchise,rights_holder,subcategory,region,role,habitat
```

The runner uses `prompt`, `target_concept`, `id`, and `category`. Other columns are stored as prompt-case metadata in the result files.

`--prompt-file` supports CSV, JSON, and JSONL:

- CSV: rows with a required `prompt` column.
- JSON: either a list of prompt objects or an object with `prompts`/`cases`.
- JSONL: one prompt object or prompt string per line.

## Example Commands

Run every available prompt case:

```powershell
python main.py --model mock --attack textfooler_style --defense clip_similarity --prompt-file data/all_prompt_cases.csv --out results/all_textfooler_clip
```

Run only one dataset:

```powershell
python main.py --model mock --attack textfooler_style --defense clip_similarity --prompt-file data/datasets/unbranding/unbranding_sample_10_per_brand.csv --out results/unbranding_textfooler_clip
```

Run an adaptive Groot batch with its local multimodal judge:

```powershell
python main.py --model diffusers --model-config configs/models/sd35_medium.yaml --attack groot --defense none --prompt-file data/datasets/representative/representative_prompt_batch_100.csv --max-candidates 5 --out results/groot_sd35_batch
```

For large diffusion models, this keeps the same model adapter alive during the run. It does not start a new Python process per prompt.
