# Copyrighted Characters 5 Per Character Sample

Manual prompt set for benign research/debug evaluation of character-name prompt defenses.

This file is not copied from an external dataset. It contains neutral text prompts written for this project, using well-known fictional character names as `target_concept` values.

Sampling rule:

- 30 fictional characters
- 5 prompts per character
- 150 total rows
- Neutral prompt templates only: portrait, scene, activity, prop, and action

The sample is stored in framework-ready CSV format:

```csv
id,prompt,target_concept,category,franchise,rights_holder,difficulty,template_id
```

Suggested use:

- Use this for local/debug experiments when you need restricted concepts such as fictional characters.
- Check your course/server policy before generating images from these prompts on shared university hardware.
- For publication-style experiments, report that this is a manually curated prompt set rather than a public benchmark dataset.
