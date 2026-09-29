# SAFREE For SDXL

This project implements SAFREE as a generation-time defense for
`stabilityai/stable-diffusion-xl-base-1.0`. The final SDXL matrix evaluates all
project attacks with and without SAFREE. The project uses its shared LLM image
evaluator rather than SAFREE's NudeNet benchmark protocol.

Paper: https://openreview.net/pdf?id=hgTFotBRKl

Authors' code: https://github.com/jaehong31/SAFREE

Reference revision: `b8b2c3fa9d7f51c46f5a570170503fc98bd9c7ec`

## What The YAML Concepts Mean

SAFREE needs to know which concept should be reduced. It converts that concept
to SDXL embeddings and constructs a concept subspace. It does not search for or
replace the literal phrase in the attacked prompt.

The default configuration uses each prompt row's `target_concept`:

```yaml
defense:
  name: safree
  concept_source: target
  concepts: []
```

For example, if Groot changes `blue rabbit mascot` into visual attributes,
SAFREE still constructs its concept subspace from `blue rabbit mascot`. It then
detects which attacked-prompt tokens move toward that subspace.

To use the same concepts for every prompt instead:

```yaml
defense:
  name: safree
  concept_source: configured
  concepts:
    - first concept phrase
    - second concept phrase
```

Each phrase is a semantic direction, not a keyword rule.
The same phrases are also joined into SDXL's negative prompt, matching the
authors' generation setup.

## Three Stages

1. **Selective token projection:** mask each attacked-prompt token in turn,
   measure its relation to the configured concept space, and project only the
   detected trigger-token embeddings away from that space.
2. **Self-validating filtering:** compare the original and fully projected
   embeddings. A larger difference applies the projected embeddings for more
   early denoising steps.
3. **Latent re-attention:** during those filtered steps, run negative, filtered,
   and original branches together. In SDXL upsampling blocks, attenuate dominant
   low-frequency features in the filtered branch relative to the original.

The LLM evaluator remains separate. It decides whether the generated image
still satisfies the attack objective; SAFREE only changes image generation.

## Source And Deviation Map

| Part | Source followed | Project implementation |
|---|---|---|
| SDXL text features | Authors' SDXL pipeline | Uses SDXL's second tokenizer/text encoder for masked prompts and concept embeddings; replaces only the second encoder's token features. |
| Trigger detection | Paper Eqs. 2-4 and authors' `safree_projection` | Same leave-one-token-out distance rule and `alpha=0.01`. |
| Token projection | Paper Eq. 5 and authors' SDXL code | Uses the released code's `(I-PC) PI p` multiplication order. The paper prints `PI (I-PC) p`; this mismatch is retained and reported. |
| Self-validation | Paper Eq. 6 and authors' SD 1.4 `f_beta` path | Ports the released calibrated cosine-distance mapping to SDXL. The released SDXL path passes `svf` but does not use it. |
| Latent re-attention | Paper Eqs. 7-8 and authors' SD 1.4 three-branch path | Ports the three branches to SDXL and uses temporary PyTorch hooks instead of replacing Diffusers block forwards. |
| Fourier update | Paper Eq. 8 | Multiplies selected filtered-branch coefficients by the scale. The released helper substitutes a scalar value and compares only real components; this project uses magnitudes and preserves complex phase. |
| Scheduler | Authors' generator | Uses `DPMSolverMultistepScheduler`, 50 steps, and guidance 7.5 in `configs/models/sdxl.yaml`. |

The authors' released SDXL script enables projection but not the two other
stages. Their SDXL call also supplies `safree` separately while the pipeline
reads `safree_dict["safree"]`. This project passes one validated request object
and fails if any of the three stages did not run.
The released latent helper can also scale U-Net backbone channels, but its
published command uses factors `1.0` and `1.0`; those identity operations are
not reproduced here.

## Experimental Deviations And Consequences

| Dimension | Paper | Project experiment | Consequence |
|---|---|---|---|
| Main backbone | SD 1.4 is the primary benchmark; the paper also demonstrates SDXL, SD3, and video models | This implementation supports `stabilityai/stable-diffusion-xl-base-1.0` only | The project tests the requested modern SDXL setting, but does not reproduce the paper's main SD 1.4 result table or its cross-architecture claims. |
| Concepts | Fixed unsafe concepts such as nudity, plus artist-removal tasks | Each dataset row's `target_concept` defines the suppression space by default | SAFREE is adapted to heterogeneous per-row safety targets; its behavior may differ from a benchmark using one fixed concept space. |
| Attack data | I2P, P4D, Ring-A-Bell, MMA-Diffusion, UnlearnDiff, and separate artist/video datasets | The final matrix uses the project's 100-prompt non-sexual safety dataset and project attack implementations | The results measure the project's selected non-sexual policy categories and cannot be compared directly with the paper's attack-specific ASR values. |
| Safety evaluation | Nudity ASR for T2I, with NudeNet-based measurements described in the appendix; other tasks use their own protocols | The shared local multimodal judge checks policy violation, preserved intent, and confidence | The project obtains one common metric across all attacks and defenses, but changes the label source and the definition of success. |
| Generation quality | FID, CLIP, and TIFA on COCO samples; LPIPS and GPT-4o for artist removal | The final matrix reports attack success and does not reproduce those quality/utility benchmarks | A lower project ASR does not by itself prove the same image-quality preservation claimed in the paper. |
| Comparison protocol | Paper-specific defense baselines and evaluation protocol | Project defenses and shared benchmark scripts | The matrix supports internal comparison, not rank or parity claims against the paper's baseline table. |

The project results are therefore evidence about SAFREE within this framework:
the same SDXL model, dataset, attacks, and evaluator are used for defended and
undefended runs. They should not be described as a reproduction of the paper's
reported ASR or generation-quality numbers.

## Verification Scope

The automated tests cover selective projection, the self-validation mapping,
branch-local Fourier updates, temporary U-Net hooks, concept selection,
configuration validation, release only after all stages finish, and a fake full
sampler that exercises all three branches. Run them with:

```bash
pytest tests/test_safree.py
```

These tests validate the implementation's mathematics and control flow without
downloading SDXL. They do not prove tensor-by-tensor parity with the authors'
old pipelines or reproduce the paper benchmark. The final SDXL batch provides
end-to-end project evidence instead.

## Not Copied

- the authors' fork of the complete old Diffusers SDXL pipeline;
- their old pinned Python environment;
- NudeNet, benchmark datasets, and paper evaluation scripts;
- SD 1.4, SD3, and video implementations;
- artist-specific evaluation logic.

No SAFREE source file is copied verbatim. The implementation uses the paper's
equations and the pinned repository revision as behavioral references. This is
also necessary because that revision does not contain a top-level license.

## Run Groot Against SAFREE

Start the configured local LLM, then run:

```bash
python main.py \
  --model diffusers \
  --model-config configs/models/sdxl.yaml \
  --attack groot \
  --defense safree \
  --prompt "a blue rabbit mascot standing in a garden" \
  --target "blue rabbit mascot" \
  --max-candidates 3
```

For a batch, replace `--prompt` and `--target` with `--prompt-file`. Every CSV
row must contain `target_concept` while `concept_source` is `target`.

The full technical record is written to `details.jsonl` under
`image_defense.metadata`. It includes trigger tokens, masked-token distances,
self-validation distance, filtered-step count, and latent re-attention calls.
