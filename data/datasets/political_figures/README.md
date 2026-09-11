# Political Figures 5 Per Person Sample

Manual prompt set for benign research/debug evaluation of public-figure prompt defenses.

This file is not copied from an external dataset. It contains neutral text prompts written for this project, using well-known public figure names as `target_concept` values.

Sampling rule:

- 40 public figures
- 5 prompts per person
- 200 total rows
- 4 balanced subcategories:
  - `polarizing_political_figure`
  - `mainstream_political_figure`
  - `scientific_public_figure`
  - `institutional_public_figure`
- 10 people per subcategory
- 50 prompts per subcategory

The sample is stored in framework-ready CSV format:

```csv
id,prompt,target_concept,category,subcategory,region,role,difficulty,template_id
```

The prompts are neutral and avoid endorsements, scandals, fake events, violence, or persuasion.
