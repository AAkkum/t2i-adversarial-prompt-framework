# TraSCE on Stable Diffusion 1.4

TraSCE is a published inference-time defense for concept erasure. We use it as
the defense in our Ring-A-Bell experiment because it changes the diffusion
trajectory without training or changing model weights. It does not simply reject
the adversarial prompt.

The active setup is Ring-A-Bell -> TraSCE -> `CompVis/stable-diffusion-v1-4`.
The framework target is the experiment's concept, for example dog. Ring-A-Bell
uses it as its attack target; TraSCE uses it as the concept to suppress.

## Sources and attribution

- Anubhav Jain et al., *TraSCE: Trajectory Steering for Concept Erasure*.
- [Paper](https://arxiv.org/abs/2412.07658),
  [audited v2](https://arxiv.org/html/2412.07658v2).
- [Official repository](https://github.com/SonyResearch/TraSCE).
- Reference commit: `244cece1f3aa96a82021e461752bd310fd0c1d67`.
- Reference implementation: `generate_images_concept_erasure.py`, particularly
  `sample()` (lines 198-315), and the README's experiment commands.

The loss, guidance combination and update order are adapted from that executable
code. Integration with the framework is project code. The original Apache 2.0
license is retained in [licenses/trasce.txt](licenses/trasce.txt). This is not a
claim that we independently invented the method.

## How it works

At the same current latent x and timestep t, the SD1.4 UNet predicts noise under
three conditions:

- u: the empty text, encoded by CLIP. This is not a zero embedding.
- p: the adversarial prompt returned by Ring-A-Bell.
- n: the concept text supplied as the negative condition, for example dog.

The CLIP tokenizer/text encoder uses all 77 positions and the final hidden states.
There is no pooling or clip-skip. All model computations in the primary preset
use FP32. Let a be guidance_loss_scale, b be discriminator_guidance_scale and s
be guidance_scale. Each step computes:

```text
d = ||p - n||_2
L = -a * exp(-d / sigma)
g = gradient of L with respect to x
x_steered = x - b * g
prediction = u + s * (p - n)
x_next = DDIM.step(prediction, t, x_steered, eta=0)
```

The norm is global over the prediction tensor (one 4 x 64 x 64 sample at 512 x
512), not an average or RMS distance. Both p and n retain their dependence on x.
Predictions are not recomputed after steering. Each new timestep starts from a
detached latent, and no weight gradients or optimizer steps are used.
The discriminator parameter name does not mean a discriminator model is loaded.

`concept_erasure=null` resolves to the unchanged framework target.
`negative_prompt=null` resolves to that concept text. An explicit negative prompt
takes precedence. The published Object example uses an object name and allows
other object names, so dog is a valid condition; no prescribed sentence or synonym
list is required for objects. This follows the author's full negative-prompt
branch, not the separate branch where only concept_erasure is supplied.

## Defaults and scheduler

| Setting | Value | Source |
|---|---|---|
| Model | Stable Diffusion 1.4 | Author script |
| Prediction | epsilon | SD1.4/DDIM config |
| Scheduler | DDIM, eta=0 | Active author sampler |
| Steps | 50 | Author script |
| Guidance | 7.5 | Author default without CSV override |
| Image size | 512 x 512 | Author script |
| Precision | FP32 | Author loading path |
| CPU offload / checkpointing | Enabled | Project engineering |

The SD1.4 config normally declares PNDM. `configs/models/sd14.yaml` explicitly
selects DDIM for both none and TraSCE. Construction preserves the model's beta
schedule: scaled_linear, beta_start=0.00085, beta_end=0.012, 1000 training steps,
steps_offset=1, set_alpha_to_one=false and clip_sample=false. Epsilon prediction,
leading timestep spacing, no thresholding and no zero-SNR rescaling are used.
The 50 timesteps are 981, 961, ..., 21, 1. Initial noise has shape [1,4,64,64],
standard normal distribution and init_noise_sigma=1. No extra scaling is added.

Category settings come from the published README commands:

| Category | guidance_loss_scale | discriminator_guidance_scale | sigma |
|---|---:|---:|---:|
| object (default) | 1 | 1 | 1 |
| style | 1 | 1 | 0.25 |
| nudity | 1.5 | 1 | 2 |
| violence | 1.5 | 1 | 1 |

Categories are explicit, not inferred from target text. Null scalars in
`configs/defenses/trasce.yaml` select the category values. Raw parser fallbacks
15/5/1 are not the published experiment-command settings.

## Paper versus executable code

Algorithm 1 uses `-exp(-||p-n||^2/(2*sigma^2))`. The executable script uses the
unsquared L2 exponential shown above. The paper's Object experiment states
lambda=1 and sigma=sqrt(0.5); the executable README Object command uses 1/1/1
with its different kernel. We follow the pinned executable code, not a mixture
of these formulas. The actual scheduler is DDIM despite an unused DPM parser
argument. The concrete scheduler update also follows the code rather than a
literal transcription of Algorithm 1's displayed recursion.

## Running the comparison

Real Diffusers runs use the shared local LLM evaluator from main. Start its
server with `./scripts/start-local-llm.ps1` (PowerShell) or
`scripts/start-local-llm.sh` before running the command below. Server and judge
settings are in `configs/local_llm.yaml` and `configs/evaluation/llm_judge.yaml`.
The attack and defense do not replace or bypass this evaluator.

Use the existing CUDA environment with `pip install -e ".[models]"` if needed.
The full defense command is:

```powershell
python main.py --model diffusers --model-config configs/models/sd14.yaml --attack ring_a_bell --defense trasce --prompt "a dog sitting in a park" --target "dog" --seed 42 --out ("results/rab_sd14_trasce_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff"))
```

For the baseline, use `--defense none` and a different output folder, keeping
all other settings unchanged. Both paths use the same FP32 initial draw, CPU
seeded generator under offload, DDIM schedule, image size, 50 steps and guidance
7.5. Their native initial latents are checked for equality in a small pipeline
test. The model preset disables the bundled post-generation safety checker for
both paths, matching the author sampler. Existing configurations keep their
previous checker behavior. No old custom defense is combined with TraSCE.

This compares the full defense: none uses `u+s*(p-u)`, whereas TraSCE uses
`u+s*(p-n)` AND a latent update. It is not a gradient-only ablation. TraSCE's
settings determine its steps and guidance; keep model and defense settings equal
when overriding defaults for a controlled comparison.

Diagnostics report loss, prediction distance, gradient norm, actual steering,
latent norm, relative steering and finite status. Values are computed every step;
console output shows steps 1, 5, 10, ... and the last step, plus min/max/mean.
Only detached scalars are retained. These are technical diagnostics, not content
metrics. The compact result files contain the shared judge decision; `details.jsonl`
contains the full attack and defense metadata. Resolved settings are in `config.yaml`;
console summaries are not an independent erasure assessment.

## What concept erasure means here

TraSCE steers the diffusion trajectory away from the erased concept. It is not
an image-space masking, inpainting or blur operation. In Appendix 7, the authors
measure Object Erasure with a pretrained ResNet50 on Imagenette classes: low
recognition of the erased class and retained recognition of other classes.
This is a classifier-based operational measure, not proof that every visual
trace is absent. The paper also notes classifier bias and the possibility of
steering toward a secondary class. It does not prescribe a blurred result or a
particular replacement object.

The user observed a clear dog without the defense and a strongly changed,
deformed central structure with target dog that was no longer clearly recognizable
as a dog to an uninformed viewer. This is a qualitative observation of one sanity
test. Deformation or ambiguity can occur in a generated result, but neither is
the definition of TraSCE nor sufficient evidence of successful erasure. Image
quality and concept recognition need separate assessment by the evaluation team.
This defense implements no additional content evaluator. The existing shared LLM
judge evaluates the generated image and controls experiment success, unchanged.
Defense bypass flags describe passage through the pipeline, not proof that the
target survived TraSCE.

## Deviations from the original implementation

| Status | Original | Ours | Reason | Likely impact |
|---|---|---|---|---|
| PAPER/CODE DIFFERENCE | Paper Gaussian kernel and Object sigma=sqrt(0.5) | Executable-code kernel and README Object sigma=1 | Follow one pinned executable reference | Must not claim exact paper-formula reproduction |
| ENGINEERING DIFFERENCE | Batched three-condition UNet call | Three sequential calls | Lower peak memory | Same algebra; floating-point rounding may differ |
| ENGINEERING DIFFERENCE | Resident GPU models, no checkpointing | CPU offload and non-reentrant checkpointing | RTX 3070 Ti 8 GB | Extra transfers/recomputation, preserved latent gradients |
| ENGINEERING DIFFERENCE | Global CUDA random state | Explicit CPU generator under offload | Matched framework baseline | Same distribution, different author noise samples |
| ENGINEERING DIFFERENCE | Standalone sampling and original libraries | Diffusers integration and installed library versions | Reuse framework model/output interfaces | Numerical differences; no bitwise reproduction claim |
| ENGINEERING DIFFERENCE | Truncation when converting pixels to uint8 | Native Diffusers rounding | Shared image output path | At most a quantization-level difference for identical continuous pixels |
| PROJECT DATA DIFFERENCE | Published benchmarks and object classes | Project Ring-A-Bell pairs/prompts and dog sanity test | University experiment | No direct comparison with published erasure scores |

Model, conditioning roles, guidance, executable loss, gradient sign, update order,
DDIM parameters and normal defaults are MATCH. No METHOD MISMATCH was found.
This is a paper-near implementation on SD1.4 with the differences listed above.

## Integration and limits

The defense stores its request in the existing shared context. A small model-
adapter hook runs it during generation; evaluation/runner.py is untouched.
The CLI rejects incompatible models before expensive attack search. TraSCE
supports SD1.4/DDIM/FP32 only; no fallback is used. SD3.5 remains generally available
in the framework. An earlier technically executable SD3.5 Flow/MMDiT experiment
was retired because it changed the prediction representation and scheduler.
No active flow adaptation remains.

The UNet's flags, mode and checkpoint state are restored after generation or
failure. Encoding and decoding use no_grad, and the attack encoder is released
before image generation. These optimizations do not reduce steps or remove
conditioning branches. One completed hardware run is not a guarantee of memory
fit in every environment. OOM is reported without an undefended replacement image.

`tests/test_trasce.py` covers the equations, gradient through both conditions,
immutable weights, checkpoint/batch parity, scheduler inputs, initialization,
configuration and backend rejection. Tiny models are **TECHNICAL SMOKE TEST /
NOT PAPER-COMPARABLE**. They cannot establish real-image erasure. Record package
versions, model revisions, seed and configs for later experiments; dependencies
are not a fully pinned reproduction environment. A generated image or existing
framework success flag alone is not evidence of erasure or attack success.
