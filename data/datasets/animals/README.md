# Animal Prompt Batch

Framework-ready prompt cases for recognizable animals.

- `animal_prompt_batch_300.csv`
  - 300 rows
  - 100 animal targets
  - 3 prompts per animal
  - covers land, water, flying, reptile/amphibian, and insect cases

Columns:

```csv
id,prompt,target_concept,category,habitat,template_id
```

Example:

```powershell
python main.py --model mock --attack textfooler_style --defense clip_similarity --prompt-file data/datasets/animals/animal_prompt_batch_300.csv --out results/animals_textfooler_clip
```
