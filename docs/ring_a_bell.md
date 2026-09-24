# Ring-A-Bell

Ring-A-Bell searches for an adversarial text prompt using a frozen text encoder.
We use it to test a published attack against TraSCE. The attack itself does not
query the image generator or the defense during search.

Our experiment is: original prompt + target -> Ring-A-Bell -> adversarial prompt
-> TraSCE (or none) -> Stable Diffusion 1.4. The framework target always means the
experiment's target concept. It is not a replacement prompt or a new label.

## Sources and attribution

- Hsu, Tsai et al., *Ring-A-Bell! How Reliable are Concept Removal Methods for
  Diffusion Models?*, ICLR 2024.
- [Paper](https://openreview.net/forum?id=lm7MRcsFiS),
  [authors' version](https://arxiv.org/html/2310.10012v3).
- [Official repository](https://github.com/chiayi-hsu/Ring-A-Bell).
- Reference commit: `e4585ada0a5eb185fbef89bb94147c97f8d2f79a`.
- Reference code: `Get_Concept_Vector.ipynb` and `InversePrompt.ipynb` at that commit.

The concept extraction and search operations are adapted from these notebooks,
not claimed as independently invented code. The authors' MIT license and
copyright notice are retained in [licenses/ring_a_bell.txt](licenses/ring_a_bell.txt).
The framework integration and concept-pair generator are project code.

## How the attack works

1. Load the CLIP tokenizer and CLIPTextModel from
   `CompVis/stable-diffusion-v1-4`. Only its text components are needed for search.
2. Encode positive and negative descriptions for the target. Use the full
   77 x 768 final hidden state, including padding positions. There is no pooling,
   attention-mask exclusion or embedding normalization.
3. Compute the concept vector as the mean of positive minus negative embeddings.
   Five repetitions per example and NumPy aggregation follow the notebook.
4. Form `target_embedding = encode(original_prompt) + coefficient * concept_vector`.
5. Search token sequences with the notebook's genetic operators. Fitness is
   `sum((target_embedding - candidate_embedding)**2)` over all 77 x 768 entries.
6. Decode the final generation's best candidate and return its text to the framework.
   Release the attack encoder before loading the image model.

A sequence contains BOS 49406, 16 mutable tokens, and EOS 49407 padding to 77
positions. The search samples token IDs 1 through 49405. It keeps the best 100
parents, performs the authors' one-point crossover and mutates one position in
selected sequences. Population size can vary after crossover; parents can also
mutate. We do not add a protected elite, refill the population or replace the final
winner with a separate best-ever archive. Only the mutable token positions are
decoded. The resulting text can contain unusual words or replacement characters;
it is passed on unchanged rather than manually cleaned up.

## Defaults

These values come from the executable `InversePrompt.ipynb`, rather than a claim
that the paper uses one setting for all concepts or experiments.

| Config field | Value |
|---|---:|
| population_size | 200 |
| generations | 3000 |
| mutation_rate | 0.25 |
| crossover_rate | 0.5 |
| coefficient | 3 |
| prompt_length | 16 mutable tokens |
| batch_size | 32 (project memory setting) |
| log_interval | 50 (project console setting) |

See `configs/attacks/ring_a_bell.yaml`. Search uses local seeded Python, NumPy and
PyTorch generators. The same environment and inputs are reproducible; different
library versions or hardware can still change floating-point results and ranking.

## Project-created concept pairs

`data/ring_a_bell/concept_pairs.json` contains **PROJECT-CREATED CONCEPT PAIRS**:
101 concepts, 12 pairs each, 1,212 pairs total. These cover the 100 distinct targets
in the representative CSV plus red. None of these pairs, including red, is claimed
to be an original author dataset.

The source is `concept_pair_spec.json`: 12 context templates, a positive subject
and three reviewed negative subjects for each CSV concept. Each negative is used
in four contexts. The existing red pairs are stored explicitly in the spec.
The script builds both the pair file and `representative_target_audit.csv`.
It pins the source CSV hash and never rewrites the dataset.

Verify the checked-in data without rewriting it:

```powershell
python scripts/build_ring_a_bell_concept_pairs.py --check
```

To rebuild after a reviewed spec change, run the same script without `--check`.
Generated outputs should be reviewed before using them in an experiment.

Concept lookup applies Unicode NFKC, casefold and whitespace normalization only.
There is no fuzzy or substring matching. Donald Trump and Barack Obama remain
separate concepts. The original prompt and framework target are not normalized
or relabeled globally.

Controlled subject substitutions reduce context changes, but cannot guarantee
perfect semantic isolation. Person identity, brands, characters and compound
concepts are particularly difficult. The pairs were not validated by image
classifiers. Template repetition and negative-subject choice can bias a vector.
The representative audit records 93 explicit targets and seven unbranding cases
requiring semantic review (six ambiguous, one conflicting cue). In particular,
the Audi case describes a twin-kidney grille, a conflicting brand cue. These cases
are documented in the audit CSV and source spec; the source dataset is unchanged.

## Running the attack

Real Diffusers runs use the shared local LLM evaluator from main. Start its
server with `./scripts/start-local-llm.ps1` (PowerShell) or
`scripts/start-local-llm.sh` before running the command below. Server and judge
settings are in `configs/local_llm.yaml` and `configs/evaluation/llm_judge.yaml`.
The attack and defense do not replace or bypass this evaluator.

Use the existing CUDA environment and install the model extras if needed:
`pip install -e ".[models]"`. The primary image config is `configs/models/sd14.yaml`.
It uses FP32, CPU offloading, DDIM, 50 steps, guidance 7.5 and 512 x 512 images.
The baseline command is a full search, not a smoke test:

```powershell
python main.py --model diffusers --model-config configs/models/sd14.yaml --attack ring_a_bell --defense none --prompt "a dog sitting in a park" --target "dog" --seed 42 --out ("results/rab_sd14_none_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff"))
```

Use the same model config with `--defense trasce` for the defense experiment;
see [TraSCE](trasce.md). SD3.5 remains available elsewhere in the framework, but
it is not the selected backend for this experiment.

The terminal reports search progress and the discovered text. Existing framework
outputs include the resolved config, compact result CSV/JSONL, `details.jsonl`
and generated images. Attack metadata in `details.jsonl` records parameters,
token IDs, fitness and concept-data hashes.
Fitness measures text-embedding distance; neither it nor a generated image proves
attack success. Content evaluation is handled by other team members.

## Deviations from the original implementation

| Status | Original | Ours | Reason | Likely impact |
|---|---|---|---|---|
| PROJECT DATA DIFFERENCE | Author nudity/violence pairs and benchmark subsets | 101 project concepts, 12 pairs each, representative and manual prompts | Project scope | Different concept vectors and difficulty; no direct ASR comparison |
| ENGINEERING DIFFERENCE | Notebook-global random state | Explicit per-run generators | Reproducibility without changing global state | Same search rules, different random sequence from an unseeded notebook |
| ENGINEERING DIFFERENCE | Whole-population encoding on CUDA | Batches of 32, optional CPU, explicit FP32/eager attention | Memory use and representation control | Floating-point ranking can differ across backends |
| ENGINEERING DIFFERENCE | Standalone notebooks | AttackCandidate, lazy loading, validation, cleanup and metadata | Framework integration | Does not add image-model feedback to search |
| PROJECT DATA DIFFERENCE | Original removal models, online services and benchmark protocol, including Union experiments | One notebook-style search with SD1.4 and TraSCE or none | Our selected comparison | New experiment; no reproduction of Union or original reported results |

The encoder representation, target formula, token range, fitness and genetic
operators are MATCH. No METHOD MISMATCH was found in the final code audit.
One notebook-style search is implemented; the paper's Union experiments and
service-specific prompt processing are not reproduced. We make no Union-result
claim. This is a paper-near reimplementation with the differences above, not an
exact reproduction of the authors' experiments.

## Tests and limits

`tests/test_ring_a_bell.py` checks search operators, defaults, deterministic mocked
search, resource cleanup and integration. `tests/test_ring_a_bell_concept_data.py`
checks reproducible data generation, coverage, normalization and data-quality
constraints. Small settings in these tests are **TECHNICAL SMOKE TEST / NOT
PAPER-COMPARABLE**; they do not change the normal defaults.

For strict reproduction, record the resolved model revisions, package versions,
seed and input hashes. The base dependency file does not pin a full environment.
The encoder is model-specific even though the attack does not query the target
image model. Successful text search alone does not guarantee retained image content.
