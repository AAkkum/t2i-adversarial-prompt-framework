# DACA

`daca` implements the ontology-guided multi-agent workflow from *Harnessing
LLM to Attack LLM-Guarded Text-to-Image Models* (arXiv:2312.07130v4).

## Reference Assets

The helper-prompt JSON files under `data/daca/reference/` are copied from the
official repository at commit `2dff91882c23c7ff054180e4a87eb09e77b2f978`.
They retain the authors' six decomposer specifications, four polisher
specifications, four assembler specifications, output formats, and one-shot
demonstrations.

## Implementation Modes

### `paper_algorithm`

This is the default research mode. It uses the released helper prompts but
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

Using SD 3.5, local defenses, or the project's representative dataset measures
transfer of DACA's method. It does not reproduce the paper's reported DALL-E 3
or Midjourney bypass rates. Reports must state the attack LLM, target image
model, defense, dataset, candidate count, and implementation mode.

Reference paper: https://arxiv.org/abs/2312.07130

Official implementation: https://github.com/researchcode001/daca
