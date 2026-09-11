# Data

This folder contains prompt datasets and framework support data.

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

## Support Files

- `restricted_concepts.yaml`  
  Restricted concepts and aliases used by defenses.

- `attack_terms.yaml`  
  Shared word lists used by attacks/paraphrasers.

- `example_prompts.csv`  
  Tiny smoke-test CSV.

## Combined CSV Schema

`all_prompt_cases.csv` keeps the framework-required columns first:

```csv
id,prompt,target_concept,category,dataset
```

Extra metadata columns are kept when available:

```csv
difficulty,template_id,prompt_set,seed,source_filename,caption_variant,image_file,source_file,franchise,rights_holder,subcategory,region,role
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

Run a batch with Groot-lite, image CLIP, and prompt-prompt similarity:

```powershell
python main.py --model diffusers --model-config configs/models/sdxl.yaml --attack groot_lite --config configs/sdxl_clip_eval.yaml --defense none --prompt-file data/datasets/synthetic/synthetic_prompt_batch_300.csv --max-candidates 1 --out results/groot_sdxl_synthetic_batch
```

For large diffusion models, this keeps the same model adapter alive during the run. It does not start a new Python process per prompt.
