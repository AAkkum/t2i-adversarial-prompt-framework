# Representative Prompt Batch 100

`representative_prompt_batch_100.csv` is a compact benchmark batch for repeated
attack/defense combinations.

It contains exactly 100 rows with 100 unique `target_concept` values:

- 20 synthetic prompts
- 20 animal prompts
- 12 unbranding prompts
- 16 copyrighted-character prompts
- 16 political/public-figure prompts
- 16 celebcaption public-figure prompts

Use this file when a full dataset run would be too expensive, for example when
manually running many attack/defense combinations.

Example:

```bash
python main.py \
  --model mock \
  --attack groot_lite \
  --defense none \
  --prompt-file data/datasets/representative/representative_prompt_batch_100.csv \
  --max-candidates 1
```

For Groot-lite, the matching decomposition template is:

```text
data/groot_decompositions_representative_100_template.yaml
```

Copy/fill that template when you want Groot-lite to rewrite all 100 targets.
