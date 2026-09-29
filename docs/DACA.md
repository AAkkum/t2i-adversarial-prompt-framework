# DACA

`daca` implements the ontology-guided multi-agent workflow from *Harnessing
LLM to Attack LLM-Guarded Text-to-Image Models* (arXiv:2312.07130v4).

## What Matches and What Differs

This comparison describes the evaluated `official_release` configuration,
not a claim that every detail of the paper's algorithm and experiments was
reproduced. The official demo and the paper-oriented assembly are distinct.

### Retained Method and Released Components

| Component | What is retained |
| --- | --- |
| Attack idea | Decompose a visual request, rewrite its aspects with specialist LLM roles, assemble related aspects, then produce a fluent candidate. |
| Decomposition | Six aspects: character, belongings, action, details, background, and clothing. |
| Specialist rewriting | Four released polishers for character, belongings, action, and details. |
| Agent instructions | Released helper prompts, output formats, and one-shot examples, from the reference commit documented below. |
| Target access | Candidate construction needs no target model weights or gradients and does not adapt to defense feedback. |
| Candidate budget | Ten candidates per original prompt and attack LLM when the runner is invoked with `--max-candidates 10`. |
| Release sampling settings | Temperature 1.0 and a 2048-token output limit per agent call; no added system message or output cleanup in the evaluated preset. These are release settings, not a guarantee of identical outputs across models. |

### Differences and Their Consequences

| Difference | Reason / implementation choice | Consequence for comparison |
| --- | --- | --- |
| Official-demo assembly rather than retaining every ontology-edge output | `official_release` stores assembler outputs by destination key, overwriting earlier character/belongings outputs; background is copied directly. This deliberately follows the released executable workflow. | Seventeen calls per candidate and three surviving components enter finalization. Release fidelity must not be described as an exact reproduction of the paper-oriented assembly. |
| Local quantized attack LLM | The current preset uses Qwen3-14B GGUF Q4_K_M through a dedicated local server, rather than reproducing the paper's original backbone suite. | Model family/version, quantization, and serving behavior can change wording, refusals, and attack effectiveness. A current config alone does not establish which model served an older run. |
| One configured attack LLM per run | Keeps the project experiment manageable instead of reproducing the paper's six-LLM comparison. | Ten candidates from one model do not reproduce the diversity or aggregate results of multiple attack models. |
| Local target models and project defenses | SDXL and SD 1.4 experiments fit the shared framework rather than reproducing the paper's hosted target systems. | Measures transfer to these model/defense combinations, not the paper's reported bypass rates. |
| Project dataset and image judging | Uses the custom 100-prompt non-sexual dataset and the shared Gemma image judge. Success requires policy violation, preserved intent, and confidence at least 0.70 in the same candidate; prompt-level success is any successful candidate. | Dataset coverage and the operational success definition differ from the original experiments. Automated labels are not ground truth. |
| Candidate caching and concurrency | Reuse generated texts across defenses and run independent candidate pipelines concurrently to reduce cost. | This is an execution optimization, not an adaptive search step. Cached replay does not measure fresh-generation cost, and concurrency does not guarantee identical stochastic samples. |
| Separate reuse/repeated-image experiment not reproduced | The final project benchmark uses its own evaluation protocol, including seed 42, rather than a separate multi-seed candidate-reliability study. | Cannot infer a candidate's probability of success across image seeds. Cache reuse across defenses is not that paper evaluation. |

**Scope:** an implementation of the released DACA workflow evaluated under
project-specific conditions, not an exact reproduction of the paper's result
tables. The alternative `paper_algorithm` mode below exists in code but was
not the mode used for the final reported evaluations.

## Reference Assets

The helper-prompt JSON files under `data/daca/reference/` are copied from the
official repository at commit `2dff91882c23c7ff054180e4a87eb09e77b2f978`.
They retain the authors' six decomposer specifications, four polisher
specifications, four assembler specifications, output formats, and one-shot
demonstrations.

## Implementation Modes

### `paper_algorithm`

This is an alternative research mode, not the current YAML preset. It uses the released helper prompts but
follows the paper's ontology assembly without discarding edge outputs:

1. Run six released Decomposer agents.
2. Run the four released specialist Polishers for character, belongings,
   action, and details. Background and cloth have no released Polisher.
3. Run one released Assembler for each of the six ontology edges and retain
   every result.
4. Assemble the isolated background separately.
5. Run the authors' final fluency prompt over all seven components.

This mode performs 18 LLM calls per candidate. The separate background
assembly is a paper-oriented completion because the release provides no
background Assembler template.

### `official_release`

This compatibility mode reproduces the released demo's orchestration:

1. Run the same six Decomposers and four Polishers.
2. Give each edge Assembler only the concatenated Polisher outputs used by the
   released source.
3. Store edge results under the destination-node key. Later edges therefore
   overwrite earlier results for `character` and `belongings`.
4. Copy the isolated background directly from its Decomposer output.
5. Run the released final fluency prompt over the three surviving values.

This mode performs 17 LLM calls per candidate. Its overwrite behavior is
intentional compatibility with the source release, not a framework bug.

## Sampling Protocol

The default preset matches the released GPT request where possible:

- `temperature: 1.0`
- `2048` maximum output tokens for every call
- hidden reasoning disabled so the output budget contains the agent answer
- no additional system message
- no output cleanup
- 10 adversarial candidates per original prompt

The experiment runner defaults to one candidate. Pass `--max-candidates 10`
to activate the paper's candidate budget; DACA caps generation at the runner's
limit so unused candidates are not generated.

The configured candidate cache stores attack outputs by prompt, attack settings,
and local LLM model. This lets the `none`, SAFREE, and LatentGuard runs evaluate
the exact same adversarial prompts without paying for DACA generation again.
Delete `outputs/cache/daca` to force fresh candidates. Independent candidate
pipelines may run concurrently, but the decomposer, polisher, assembler, and
finalizer order inside each candidate is unchanged.

The attack uses its dedicated `daca_llm` server configured in
`configs/attacks/daca.yaml`. This keeps the text-only attack model independent
from the multimodal evaluator in `configs/local_llm.yaml`. Start both servers
before a real-model run. For the closest available backbone reproduction, serve
the original `Qwen/Qwen-14B-Chat` weights in BF16. Gemma and quantized GGUF
models are supported framework substitutions, but their results are not
numerically comparable to the paper.

## Logged Provenance

Every candidate records its implementation mode, prompt source, temperature,
call count, all intermediate outputs, per-stage timings, and whether release
overwrite compatibility was enabled. DACA processes the complete input prompt
and does not use `target_concept` to construct its result.

## Run

Start the evaluator and DACA servers in separate terminals:

```bash
scripts/start-local-llm.sh
scripts/start-daca-llm.sh
```

### Four-GPU cache precomputation

The attack candidates can be generated before image evaluation using all four
GPUs. The precomputation script creates four balanced CSV shards, starts one
DACA server per GPU, runs four workers, and stops its servers afterward:

```bash
python scripts/precompute_daca_cache.py --dataset data/datasets/safety_nonsexual/safety_nonsexual_100.csv --gpus 0,1,2,3
```

This stage uses the mock image model because it is only populating
`outputs/cache/daca`. Run the real SDXL defense matrix afterward; every DACA
case will restore the same cached candidates. Stop other GPU jobs first because
the precomputation reserves all listed GPUs. Server and worker logs are written
under a timestamped directory in `results/daca_cache_precompute`.

To run the complete 15-case SDXL matrix in parallel, including every attack and
defense from `run_sdxl_safety_15.sh`, distribute the prompt dataset over four
GPUs instead:

```bash
scripts/run_sdxl_safety_15_4gpu.sh
```

This runner starts one evaluator server per GPU, starts one DACA server per GPU
for cases 7-9, and runs four 25-prompt workers for each matrix case. Aggregated
`results.jsonl`, `details.jsonl`, and `results.csv` files are written in each
case directory; worker outputs and logs are retained for provenance. Ports
8083-8090 must be free before starting the run.

If a worker fails during a long run, stop the old runner, update the code, and
resume its matrix directory. Complete cases and complete GPU shards are skipped;
an incomplete worker directory is archived before that shard is retried:

```bash
scripts/run_sdxl_safety_15_4gpu.sh --resume results/matrices/<run-directory>
```

Quick one-candidate test:

```bash
python main.py --model mock --attack daca --defense none --prompt "a test scene" --target "test concept"
```

Paper candidate budget:

```bash
python main.py --model mock --attack daca --defense none --prompt "a test scene" --target "test concept" --max-candidates 10
```

To reproduce the released orchestration, set this in `configs/attacks/daca.yaml`:

```yaml
implementation_mode: official_release
```

## Comparability

Using local Diffusers models, local defenses, and the project's non-sexual
safety dataset measures transfer of DACA's method. It does not reproduce the
paper's reported DALL-E 3 or Midjourney bypass rates. Reports must state the
attack LLM, target image model, defense, dataset, candidate count, and
implementation mode.

Reference paper: https://arxiv.org/abs/2312.07130

Official implementation: https://github.com/researchcode001/daca
