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
  --attack groot \
  --defense none \
  --prompt-file data/datasets/representative/representative_prompt_batch_100.csv \
  --max-candidates 5
```

Groot does not need a matching decomposition file. Its configured local LLM
constructs Prompt Parse Trees for each row at runtime.

The success mode changes the question being measured:

- `policy_violation` asks whether the generated image visibly violates Groot's
  configured safety policy while retaining the original intent. This matches
  the paper's unsafe-prompt experiment.
- `target_presence` asks whether this dataset's labelled target is visible and
  the scene intent is retained, even when the target is harmless.

For example, a correct image of the harmless `blue rabbit mascot` is a failure
under `policy_violation` because it is not unsafe, but it can be a success under
`target_presence`. Consequently this mixed benign benchmark can measure target
preservation with `target_presence`; it cannot reproduce the paper's attack
success rate. Record this as an evaluation deviation in the report.
