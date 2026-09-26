# SAFREE For SDXL

This project implements SAFREE as a generation-time defense for
`stabilityai/stable-diffusion-xl-base-1.0`. The initial experiment is Groot
against SAFREE; no SAFREE benchmark dataset or NudeNet evaluator is included.

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
