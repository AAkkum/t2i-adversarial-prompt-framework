# Data

Prompt files describe experiment inputs. Each row needs a `prompt` and may also
contain `target_concept`, `id`, `category`, and additional metadata.

## Recommended Batches

- `example_prompts.csv`: minimal smoke test
- `tmp_synthetic_10.csv`: cheap real-model test
- `datasets/safety_nonsexual/safety_nonsexual_100.csv`: final non-sexual safety benchmark
- `datasets/safety_nonsexual/safety_nonsexual_10.csv`: cheap safety smoke test
- `datasets/representative/representative_prompt_batch_100.csv`: benign target-preservation benchmark
- `all_prompt_cases.csv`: combined large collection

The safety benchmark contains 100 project-authored prompts across violence,
illegal activity, self-harm, shocking content, hate, and harassment. Its README
documents the methodology, limitations, and matching component data. The older
representative batch contains benign and identity-based targets and measures a
different question.

## Attack And Defense Data

- `attack_terms.yaml`: TextFooler terms
- `search_attack/`: Search Attack phrases, target descriptions, and test cases
- `latent_guard/restricted_concepts.yaml`: example protected concepts and aliases
- `latent_guard/restricted_concepts_representative_100.yaml`: fixed benchmark blacklist
- `latent_guard/restricted_concepts_safety_nonsexual_100.yaml`: safety benchmark blacklist
- `ring_a_bell/concept_pairs_safety_nonsexual_100.json`: safety benchmark concept pairs
- `latent_guard/`: location for external LatentGuard weights and parity instructions

Groot does not need decomposition files. It creates its prompt tree with the
local LLM at runtime.

## Batch Example

```bash
python main.py \
  --model mock \
  --attack identity \
  --defense none \
  --prompt-file data/datasets/safety_nonsexual/safety_nonsexual_10.csv
```
