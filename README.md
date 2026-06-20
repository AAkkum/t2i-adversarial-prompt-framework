# T2I Adversarial Prompt Framework

Safe modular framework for university experiments on prompt-based jailbreaks and adversarial prompts for text-to-image systems.

## Safety Scope

This repository is an academic evaluation framework. The default data uses harmless synthetic restricted concepts such as `blue rabbit mascot`, `red cube robot`, and `green owl emblem`. Do not add sexual, violent, illegal, or real-world misuse content. The framework is not intended to bypass deployed systems.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional model dependencies:

```bash
pip install -e ".[models]"
```

Development dependencies:

```bash
pip install -e ".[dev]"
```

## Quickstart

Single prompt with the CPU-only mock model:

```bash
python main.py --model mock --attack identity --defense none --prompt "a blue rabbit mascot standing in a garden" --target "blue rabbit mascot" --seed 42 --out results/debug_run
```

Single prompt with the normalization defense:

```bash
python main.py --model mock --attack identity --defense normalize_keywords --prompt "a blue rabbit mascot standing in a garden" --target "blue rabbit mascot" --out results/blocked_debug
```

Batch mode:

```bash
python main.py --model mock --attack identity --defense none --prompt-file data/example_prompts.csv --out results/batch_debug
```

List available components:

```bash
python main.py --list-components
```

## Interfaces

Models implement `ImageModel.generate(prompt, output_dir, seed, context)` and return a `GenerationResult`.
Attacks implement `Attack.generate(prompt, target_concept, context)` and return `AttackCandidate` objects.
Defenses implement `check_prompt(...)` and/or `check_image(...)`, returning `DefenseDecision`.

## Adding Attacks

Create a class in `t2i_framework/attacks/`, subclass `Attack`, return one or more `AttackCandidate` objects, then register it in `t2i_framework/core/registry.py`. See `docs/HOW_TO_ADD_ATTACK.md`.

## Results

Each run writes:

- `results.jsonl`: append-only machine-readable results
- `results.csv`: tabular copy of results
- `config.yaml`: run configuration snapshot
- `images/*.png`: generated images when generation was allowed

## Current Limitations

The default `success` metric only checks that the prompt and image were not blocked and that an image file exists. Real target-concept detection, CLIP scoring, and research-grade attacks are left as extension tasks.
