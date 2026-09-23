# Next Defense Paper

The current project has two defenses: `character_filter` and
`latent_guard_lite`. A third defense can be added for the final comparison.

## Recommended: Safe Latent Diffusion

Patrick Schramowski et al., *Safe Latent Diffusion: Mitigating Inappropriate
Degeneration in Diffusion Models*, CVPR 2023.

- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Schramowski_Safe_Latent_Diffusion_Mitigating_Inappropriate_Degeneration_in_Diffusion_Models_CVPR_2023_paper.html
- Code: https://github.com/ml-research/safe-latent-diffusion
- Benchmark: https://huggingface.co/datasets/AIML-TUDA/i2p

SLD changes the denoising direction during image generation so that configured
unsafe concepts are suppressed. It needs no new training and has published
weak, medium, strong, and maximum parameter settings.

The faithful implementation target is Stable Diffusion 1.4 or 1.5. Applying it
to SD 3.5 would be an adaptation because SD 3.5 has a different denoiser and
text-encoder structure.

## Alternatives

- SAFREE supports newer diffusion families but requires a larger custom-pipeline
  port: https://github.com/jaehong31/SAFREE
- GuardT2I is a prompt-stage defense but requires released decoder checkpoints
  and architecture-specific embedding dimensions:
  https://github.com/cure-lab/GuardT2I

For a one-week student implementation, SLD on SD 1.5 is the clearest paper
reproduction.
