# Defense Paper Selection

This note compares paper-based defenses that could replace Atabey's
project-specific `image_clip_filter`. No new defense has been implemented yet.

## Recommendation: Safe Latent Diffusion

**Paper:** Patrick Schramowski et al., *Safe Latent Diffusion: Mitigating
Inappropriate Degeneration in Diffusion Models*, CVPR 2023.

- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Schramowski_Safe_Latent_Diffusion_Mitigating_Inappropriate_Degeneration_in_Diffusion_Models_CVPR_2023_paper.html
- Authors' code: https://github.com/ml-research/safe-latent-diffusion
- Benchmark: https://huggingface.co/datasets/AIML-TUDA/i2p

SLD changes the denoising direction during image generation. In addition to the
normal prompt direction, it computes a direction for a configured unsafe safety
concept and subtracts unsafe influence at each applicable denoising step. Its
threshold, warm-up, guidance strength, and momentum parameters produce the
paper's weak, medium, strong, and maximum settings.

This is fundamentally different from the current image-CLIP filter. The CLIP
filter generates an image first and then accepts or deletes it based on a
target-similarity score. SLD intervenes inside Stable Diffusion so that the
unsafe visual content is less likely to be generated in the first place.

### Why It Is The Best Fit

- It is a published Stable Diffusion defense, not a project-created threshold.
- The paper, authors' code, benchmark, and parameter presets are public.
- It requires no training or new model weights.
- The algorithm and evaluation protocol are clear enough to explain and defend
  in a presentation.
- It is smaller and less fragile than reproducing a trained prompt decoder such
  as GuardT2I or porting SAFREE's custom SD3 pipeline.

### Important Boundary

The faithful and easiest reproduction should use the paper's supported Stable
Diffusion generation family, preferably SD 1.4 or 1.5. It should not be
presented as an SD 3.5 defense. SD 3.5 uses a different transformer-based
denoiser and three text encoders, so transferring the equations and parameters
would become a new adaptation rather than a direct reproduction.

The current environment's Diffusers version no longer exposes the old safe
pipeline module. Implementation should therefore either integrate the authors'
MIT-licensed `sld` package in a compatible environment or port the small safety-
guidance calculation into a maintained SD 1.x pipeline. The second option gives
the project clearer ownership of the reimplementation but needs careful tests.

### Paper-Compatible Evaluation

The paper compares base SD, a negative-prompt baseline, and four SLD strengths
on the I2P benchmark. It generates multiple images per prompt and reports the
probability of inappropriate output by combining Q16 with NudeNet. It also
checks image quality and prompt alignment so that suppressing everything is not
mistaken for a good defense.

For a manageable university reproduction:

1. Use a documented subset of I2P and fixed seeds.
2. Generate paired outputs with base SD and SLD using the same prompt and seed.
3. Evaluate unsafe-output rate with the paper's released classifiers, subject
   to ethics approval and access constraints.
4. Measure prompt preservation with CLIP only as an auxiliary quality metric,
   not as the defense itself.
5. Manually review a small labelled sample because the paper also documents
   classifier false positives.

## Alternative: SAFREE

**Paper:** Jaehong Yoon et al., *SAFREE: Training-Free and Adaptive Guard for
Safe Text-to-Image and Video Generation*, ICLR 2025.

- Project and paper: https://safree-safe-t2i-t2v.github.io/
- Authors' code: https://github.com/jaehong31/SAFREE

SAFREE removes unsafe directions from token embeddings, validates when the
filter should be active, and changes attention in the image latent space. The
paper includes SDXL and SD-V3 experiments, making it closer to the desired SD
3.5 setting.

It is not the easy option. The released SD-V3 path targets SD3 Medium with a
custom pipeline based on Diffusers 0.29, while this project uses a newer
Diffusers version and SD 3.5 Large. The repository's SD-V3 README also marks its
evaluation scripts as unfinished. Porting it would be a medium-to-large task and
the result would still need to be disclosed as an SD3-to-SD3.5 adaptation.

## Alternative: GuardT2I

**Paper:** Yijun Yang et al., *GuardT2I: Defending Text-to-Image Models from
Adversarial Prompts*, NeurIPS 2024.

- Paper: https://arxiv.org/abs/2403.01446
- Authors' code: https://github.com/cure-lab/GuardT2I

GuardT2I decodes a T2I model's text-guidance embedding back into natural
language and compares the decoded meaning with the input. It is a valid prompt-
stage defense, but it needs released trained decoder checkpoints and is tied to
the original CLIP embedding dimensions and training setup. The released
evaluation script contains hardcoded paths and assumptions. Adapting it to SD
3.5's multiple text encoders would be harder and less faithful than SLD.

## Decision

Use **SLD with SD 1.5** if the supervisor values a clear paper reproduction and
reasonable implementation risk. Use **SAFREE** only if applying the defense to
SD3/SD3.5 is a hard requirement and there is enough time to port and validate a
custom denoising pipeline. Do not describe the current image-CLIP filter as a
paper defense.
