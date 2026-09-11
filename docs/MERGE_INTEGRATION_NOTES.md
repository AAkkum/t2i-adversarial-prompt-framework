# Merge Integration Notes

Date: 2026-09-11

Integration commit on `main`: `2a7fb85 Integrate contributor attacks and evaluation updates`

This was an integration commit, not a normal Git merge commit. The contributor branches changed several shared files in incompatible ways, especially `main.py`, `t2i_framework/evaluation/runner.py`, `t2i_framework/evaluation/metrics.py`, registries, and tests. Instead of accepting one branch's architecture over the others, the useful isolated work was copied in and the shared framework code was reconciled manually.

## Branches Reviewed

- `origin/Burak`
- `origin/hans`
- `origin/abdel`

`origin/main` had not moved when the integration work started.

## Accepted From Burak

- Added the data-driven `search_attack` implementation.
- Added search attack support data under `data/search_attack/`.
- Added `character_filter` defense.
- Added `filter_placeholder` defense.
- Added BLIP captioning support for image-stage defense.
- Added MiniLM semantic concept matching support.
- Added search attack docs and tests.
- Adopted the safer image-release behavior:
  - generated images are first created in a temporary quarantine directory;
  - blocked images are not published into the result directory;
  - image-defense errors fail closed;
  - allowed images are published without overwriting old images;
  - search runs can mark a best allowed candidate.

## Accepted From Hans

- Replaced the placeholder `textfooler_style` attack with the implemented TextFooler-style attack.
- Added Ollama/Qwen paraphraser support.
- Added Ollama-based similarity judge support.
- Added CLIP text similarity support for attack candidate filtering.
- Added `clip_similarity` prompt defense.
- Added prompt-prompt similarity evaluation as an optional metric.
- Added richer prompt case loading with `case_id`, `category`, and metadata support.
- Added component config preset folders:
  - `configs/models/`
  - `configs/attacks/`
  - `configs/defenses/`
  - `configs/evaluation/`
- Added prompt datasets and related documentation.
- Added tests for config merging, prompt cases, prompt similarity, paraphrasers, judges, and word lists.

## Accepted From Abdel

- Added `pgj` attack registration.
- Added `t2i_framework/attacks/pgj.py`.
- Added `configs/models/sd35_large.yaml and configs/attacks/pgj.yaml`.

## Reshaped During Integration

The shared files were not copied wholesale from any single branch. They were rewritten into one consistent framework shape.

- `main.py`
  - Keeps the CLI generic.
  - Supports optional component config files:
    - `--model-config`
    - `--attack-config`
    - `--defense-config`
  - Auto-loads matching presets from `configs/models/`, `configs/attacks/`, and `configs/defenses/` when present.
  - Does not add attack-specific or defense-specific CLI flags.

- `t2i_framework/core/config.py`
  - Adds deep config merging.
  - Supports loading multiple YAML configs in order.

- `t2i_framework/evaluation/runner.py`
  - Keeps one shared experiment runner.
  - Supports multiple attack candidates.
  - Passes `config`, `defense`, and `max_candidates` through attack context.
  - Keeps image-text CLIP evaluation for target-preservation success.
  - Adds Hans' prompt-prompt similarity as optional auxiliary evaluation metadata.
  - Adds Burak's quarantine/release behavior for generated images.
  - Fails closed if generation or image defense raises an error.
  - Keeps one model adapter instance alive across `--prompt-file` batches.

- `t2i_framework/evaluation/metrics.py`
  - Keeps `placeholder_success`.
  - Keeps `clip_success` for image-text CLIP based attack success.
  - Adds text-overlap and score-band helper functions for auxiliary scoring.

- `t2i_framework/core/types.py`
  - Adds `PromptCase`.
  - Adds optional `case_id` and `category` fields to result rows.

## Rejected Or Not Copied Directly

- We did not make a normal merge commit from the contributor branches.
  - Reason: the branches had conflicting designs for the shared runner and config flow.

- We did not keep attack-specific logic hardcoded in `main.py`.
  - Reason: attack and defense parameters should live in YAML config files.

- We did not replace the existing image-text CLIP evaluation with prompt-prompt similarity.
  - Reason: prompt-prompt similarity measures whether the rewritten prompt stayed close to the original text, but it does not verify that the generated image contains the target concept.

- We did not let blocked images remain as final result images.
  - Reason: image-stage defenses should prevent blocked images from being published.

- We did not preserve every branch's exact runner implementation.
  - Reason: there should be one shared runner contract for all attacks and defenses.

## Verification

Before committing and pushing:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
git diff --check
```

Result:

- `98 passed`
- Ruff passed
- Whitespace check passed

## Important Git Note

GitHub may still show `Burak`, `hans`, and `abdel` as active branches. That is expected because this was not a formal merge commit and the branch refs were not deleted. Their relevant code was integrated into `main` through commit `2a7fb85`.

## Follow-up Updates

- `configs/evaluation/clip_and_prompt_similarity.yaml` enables prompt-prompt similarity in addition to image CLIP evaluation.
- `results.csv` now includes flattened `score_*` columns so metrics are visible in table form.
- `--prompt-file` now accepts CSV, JSON, and JSONL.
- `data/README.md` explains the prompt-case datasets and support-data files.
