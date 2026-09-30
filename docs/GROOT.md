# Groot

This project reimplements the main workflow from *Groot: Adversarial Testing
for Text-to-Image Generative Models with Tree-based Semantic Transformation*
(arXiv:2402.12100). The authors publish their implementation under the repository
name [TREANT](https://github.com/llm-jailbreaker/Treant).

Paper: https://arxiv.org/abs/2402.12100

Authors' code: https://github.com/llm-jailbreaker/Treant

This is a method reimplementation inside the shared project framework. It is not
an exact reproduction of the paper's experimental setup or reported results.

## Method In The Paper

The paper combines two attack strategies:

1. **Semantic decomposition:** represent a prompt as a Prompt Parse Tree (PPT)
   containing object, attribute, and relation nodes. Short phrases collected
   from the tree are shuffled into a new prompt. If the text filter still blocks
   it, an LLM selects and decomposes another sensitive leaf.
2. **Sensitive element drowning:** after the text filter is bypassed, place the
   sensitive scene on one image canvas and unrelated benign content on other
   canvases to distract the image filter.

The paper treats the target image service as a black box. It distinguishes text
and image failures from service responses, refines the prompt, and repeats until
success or a time limit. Its experiments use GPT-4, DALL-E 3, Midjourney, and
SDXL; a 300-prompt dataset covering ten prohibited categories; manual image
labels; and five repeated runs.

## Project Workflow

For each input row, this implementation:

1. Sends the original prompt as an `original_probe` without calling the LLM.
2. Stops if the shared evaluator reports success.
3. If a prompt-stage defense blocks the prompt, asks the local LLM to construct
   or deepen a PPT and renders its properties as a shuffled prompt.
4. If an image-stage defense blocks the image, or an allowed image fails the
   evaluator, creates a multi-panel drowning prompt.
5. After a failed drowning attempt, deepens one unused tree leaf and constructs
   another drowning prompt.
6. Stops on success, an unrecoverable generation error, exhaustion of the tree,
   a duplicate prompt, or the `--max-candidates` limit.

`max-candidates` counts image-generation attempts, including the original probe.
The number of local LLM decomposition calls is recorded separately.

## Code Map

| Responsibility | Implementation |
|---|---|
| Start a session and create the original probe | `GrootAttack.generate()` in `t2i_framework/attacks/groot.py` |
| React to prompt, image, and evaluator feedback | `GrootAttack.next_candidate()` |
| Store the project PPT | `PromptTreeNode` and `_GrootSession` |
| Ask the local LLM for the first tree | `_build_tree()` |
| Select and expand another leaf | `_expand_one_leaf()` |
| Flatten and shuffle tree properties | `_render_tree()` |
| Construct the multi-panel prompt | `_drowning_candidate()` and `_compose_drowning_prompt()` |
| Supply stage feedback and candidate limits | `ExperimentRunner` in `t2i_framework/evaluation/runner.py` |
| Configure tree and drowning limits | `configs/attacks/groot.yaml` |

## Source And Deviation Map

| Part | Paper | Project implementation | Consequence |
|---|---|---|---|
| LLM backend | GPT-4 constructs the PPT and helps review sensitive leaves | The configured local LLM constructs and expands the tree | Tree quality and decomposition choices can differ from the paper and depend on the selected local model. |
| PPT representation | Explicit object, attribute, and relation node types in an ordered rooted tree | Generic nodes store `properties` and `children`; root properties represent relations and children represent visible objects or elements | The important hierarchy is retained, but the complete formal grammar and explicit node-type labels are simplified. |
| Leaf selection | An LLM-based compliance review identifies a leaf for further decomposition | A seeded random order selects an untried leaf; the LLM only expands the selected leaf | Refinement is reproducible for a seed but may be less targeted toward the phrase that caused a block. |
| Prompt rendering | Collect and randomly rearrange short phrases from the PPT | Clean, flatten, and seed-shuffle node properties | This follows the paper's semantic transformation idea, but formatting and phrase order are project-specific. |
| Failure analysis | Keyword matching on black-box service error messages separates text and image failures | Explicit `prompt_blocked`, `image_blocked`, generation errors, and LLM-evaluator feedback drive the state machine | The workflow integrates cleanly with local defenses, but it does not reproduce the behavior of hosted service filters. |
| Drowning | Add unrelated canvases to distract an image safety filter | Use a configurable grid, four panels by default, with three fixed benign scenes | It implements the multi-canvas idea, but the panel wording, count, and benign content are project choices. |
| Search budget | Time-limited refinement and paper query protocol | `--max-candidates` limits image-generation attempts; tree depth and node count add hard limits | Runtime is predictable, but success and query-efficiency values are not directly comparable with the paper. |
| Ablations | Semantic Decomposition Only and Irrelevant Element Drowning Only | `semantic_only` and `drowning_only` provide analogous project modes | They isolate the same two ideas but are not exact reproductions of the paper's ablation environment. |
| Models and dataset | DALL-E 3, Midjourney, SDXL, and 300 prompts including sexual content | Local framework models and `safety_nonsexual_100.csv` for the final benchmark | The project satisfies its non-sexual safety scope, but tests a different prompt distribution and model set. |
| Success labels | Manual judgment that the generated image contains prohibited/NSFW content | Shared local multimodal judge checks policy violation, preserved intent, and confidence | All project attacks use one automated rule, but the resulting ASR cannot be compared directly with the paper's manually labelled ASR. |
| Repetition | Five runs to reduce model randomness | Seeds and configuration are recorded, but the final project matrix is not the paper's five-repeat protocol | The project records reproducible inputs but provides weaker evidence about variance across repeated generations. |

## Validity Of The Results

The implementation is suitable for comparing Groot with the other attacks and
defenses **inside this project**, because they share the same dataset, models,
runner, and evaluator. It demonstrates the paper's main attack logic: adaptive
PPT decomposition followed by multi-canvas drowning.

The percentages should not be presented as a reproduction of the paper. The
local LLM, simplified PPT, random leaf selection, fixed drowning scenes, custom
dataset, local models, candidate budget, and automated evaluator can all change
the measured success rate. These are implementation and experimental deviations,
not merely different hardware.

## Configuration

`configs/attacks/groot.yaml` controls:

- `mode`: `full`, `semantic_only`, or `drowning_only`;
- `tree.max_depth` and `tree.max_nodes`: limits for repeated decomposition;
- `tree.shuffle_phrases`: seeded phrase-order randomization;
- `drowning.panel_count` and `drowning.benign_scenes`: multi-panel layout;
- `treat_generation_errors_as_blocks`: whether infrastructure errors may trigger
  refinement. It is `false` for the final experiments so an OOM or crash is not
  mistaken for a defense block.

Choose the local model and server settings in `configs/local_llm.yaml`, then
start it with:

```bash
scripts/start-local-llm.sh
```

## Example

```bash
python main.py \
  --model diffusers \
  --model-config configs/models/sdxl.yaml \
  --attack groot \
  --defense safree \
  --prompt "a blue rabbit mascot standing in a garden" \
  --target "blue rabbit mascot" \
  --max-candidates 5
```

The final safety benchmark uses `policy_violation` in
`configs/evaluation/llm_judge.yaml`. `target_presence` asks a different,
benign target-preservation question and must not be mixed into the same table.

## Recorded Evidence

`details.jsonl` stores the strategy used for every candidate, the PPT, tree depth,
node count, drowning round, image-generation query count, and decomposition-call
count. The tests verify the control flow for the original probe, semantic
decomposition, drowning, both project ablations, and adaptive runner feedback.
They do not establish output parity with the authors' hosted-model experiments.
