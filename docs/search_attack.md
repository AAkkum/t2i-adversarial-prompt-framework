# Search Attack

`search_attack` is Burak's project-specific prompt search method. It is not a
reimplementation of one published attack paper.

## How It Works

1. Read the protected target from `--target`.
2. Replace that target with a description from
   `data/search_attack/concept_targets.json` when one is available.
3. Add controlled phrase variants from
   `data/search_attack/variant_phrases.json`.
4. Return up to `--max-candidates` prompts.
5. Let the normal defense, image model, and shared evaluator process each one.

The attack does not load the image model repeatedly. The runner keeps the model
alive for the complete prompt batch.

## Related Files

- `t2i_framework/attacks/search_attack.py`
- `t2i_framework/attacks/search_support.py`
- `configs/attacks/search_attack.yaml`
- `data/search_attack/concept_targets.json`
- `data/search_attack/variant_phrases.json`
- `data/search_attack/test_cases.json`

## Example

```bash
python main.py \
  --model diffusers \
  --model-config configs/models/sdxl.yaml \
  --attack search_attack \
  --defense character_filter \
  --prompt "Mario standing in a university laboratory" \
  --target "mario" \
  --max-candidates 10
```
