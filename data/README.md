# Data

Prompt files describe experiment inputs. Each row needs a `prompt` and may also
contain `target_concept`, `id`, `category`, and additional metadata.

## Recommended Batches

- `example_prompts.csv`: minimal smoke test
- `tmp_synthetic_10.csv`: cheap real-model test
- `datasets/representative/representative_prompt_batch_100.csv`: main compact benchmark
- `all_prompt_cases.csv`: combined large collection

The representative batch contains 100 prompts with 100 different targets from
synthetic, animal, brand, fictional-character, political-figure, and celebrity
categories.

## Attack And Defense Data

- `attack_terms.yaml`: TextFooler terms
- `search_attack/`: Search Attack phrases, target descriptions, and test cases
- `restricted_concepts.yaml`: protected concepts and aliases
- `latent_guard/`: location for external LatentGuard weights

Groot does not need decomposition files. It creates its prompt tree with the
local LLM at runtime.

## Batch Example

```bash
python main.py \
  --model mock \
  --attack identity \
  --defense none \
  --prompt-file data/tmp_synthetic_10.csv
```
