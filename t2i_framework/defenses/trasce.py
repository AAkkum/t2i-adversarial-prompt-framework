"""Paper-near, official-code TraSCE for Stable Diffusion 1.4 noise prediction.

Reference: SonyResearch/TraSCE, 244cece1f3aa96a82021e461752bd310fd0c1d67.
See docs/trasce.md for the paper/code discrepancy and adaptation limitations.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense

AUTHOR_REVISION = "244cece1f3aa96a82021e461752bd310fd0c1d67"
MODEL_ID = "CompVis/stable-diffusion-v1-4"
# Published README experiment commands, not the unused parser fallback values.
PRESETS = {
    "object": (1.0, 1.0, 1.0),
    "style": (1.0, 1.0, 0.25),
    "nudity": (1.5, 1.0, 2.0),
    "violence": (1.5, 1.0, 1.0),
}


@dataclass(frozen=True)
class TraSCESettings:
    category: str = "object"
    concept_erasure: str | None = None
    negative_prompt: str | None = None
    guidance_loss_scale: float | None = None
    discriminator_guidance_scale: float | None = None
    sigma: float | None = None
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.category not in PRESETS:
            raise ValueError(f"TraSCE category must be one of {tuple(PRESETS)}.")
        for key, default in zip(
            ("guidance_loss_scale", "discriminator_guidance_scale", "sigma"),
            PRESETS[self.category],
        ):
            if getattr(self, key) is None:
                object.__setattr__(self, key, default)
        for key in (
            "guidance_loss_scale",
            "discriminator_guidance_scale",
            "sigma",
            "guidance_scale",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{key} must be numeric.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{key} must be finite and positive; TraSCE cannot be disabled silently."
                )
        if type(self.num_inference_steps) is not int or self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be a positive integer.")
        if type(self.gradient_checkpointing) is not bool:
            raise TypeError("gradient_checkpointing must be boolean.")
        for key in ("concept_erasure", "negative_prompt"):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{key} must be non-empty text or null.")


def localized_loss(positive: Any, negative: Any, settings: TraSCESettings) -> Any:
    """Upstream code kernel: -a exp(-L2/sigma), NOT the paper's squared kernel."""
    import torch

    distance = torch.linalg.vector_norm(positive.float() - negative.float())
    return -settings.guidance_loss_scale * torch.exp(-distance / settings.sigma)


def steer_step(
    latents: Any, predict: Any, settings: TraSCESettings, diagnostics: dict | None = None
) -> tuple[Any, Any]:
    """Three condition forwards, one latent gradient, predictions at the old state.

    `predict(x, branch)` returns predicted noise; branches are empty, prompt,
    negative. Autograd must pass through BOTH non-empty branches.
    """
    import torch

    if torch.is_inference_mode_enabled():
        raise RuntimeError("TraSCE requires autograd; torch.inference_mode is incompatible.")
    with torch.no_grad():
        unconditional = predict(latents, 0).float()
    with torch.enable_grad():
        current = latents.detach().float().requires_grad_(True)
        positive = predict(current, 1).float()
        negative = predict(current, 2).float()
        loss = localized_loss(positive, negative, settings)
        gradient = torch.autograd.grad(loss, current, retain_graph=False, create_graph=False)[0]
    with torch.no_grad():
        steered = current.detach() - settings.discriminator_guidance_scale * gradient
        prediction = unconditional + settings.guidance_scale * (
            positive.detach() - negative.detach()
        )
        if diagnostics is not None:
            distance = torch.linalg.vector_norm(positive.detach() - negative.detach())
            gradient_norm = torch.linalg.vector_norm(gradient)
            steering_norm = torch.linalg.vector_norm(steered - current.detach())
            latent_norm = torch.linalg.vector_norm(current.detach())
            # Measure the actual rounded update, not merely scale * gradient.
            values = {
                "loss": loss.detach(),
                "distance": distance,
                "distance_over_sigma": distance / settings.sigma,
                "gradient_norm": gradient_norm,
                "steering_norm": steering_norm,
                "latent_norm": latent_norm,
                "relative_steering": steering_norm / latent_norm,
            }
            diagnostics.update({key: value.item() for key, value in values.items()})
            diagnostics["all_finite"] = all(math.isfinite(v) for v in diagnostics.values())
        if not torch.isfinite(steered).all() or not torch.isfinite(prediction).all():
            raise FloatingPointError("Non-finite TraSCE latent update or prediction.")
    return steered, prediction


@contextmanager
def frozen_denoiser(denoiser: Any, checkpointing: bool):
    """Restore all parameter flags, module modes and checkpoint settings on exit."""
    parameters = list(denoiser.parameters())
    flags = [p.requires_grad for p in parameters]
    modes = [(module, module.training) for module in denoiser.modules()]
    old_checkpointing = denoiser.is_gradient_checkpointing
    try:
        denoiser.requires_grad_(False)
        denoiser.eval()
        if checkpointing and not old_checkpointing:
            denoiser.enable_gradient_checkpointing()
        yield
    finally:
        if checkpointing and not old_checkpointing:
            denoiser.disable_gradient_checkpointing()
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


class TraSCEDefense(Defense):
    """Generation-time defense; prompt/image hooks do no content classification."""

    name = "trasce"

    def check_prompt(self, prompt, target_concept=None, context=None) -> DefenseDecision:
        if context is None:
            raise ValueError("TraSCE requires the shared generation context.")
        if not isinstance(target_concept, str) or not target_concept.strip():
            raise ValueError("TraSCE requires the experiment's target_concept.")
        config = dict((context.get("config") or {}).get("defense", {}))
        config.pop("name", None)
        settings = TraSCESettings(**config)
        concept = settings.concept_erasure or target_concept
        negative = settings.negative_prompt or concept
        request = {
            "target_concept": target_concept,
            "concept_erasure": concept,
            "negative_prompt": negative,
            "parameters": asdict(settings),
            "author_revision": AUTHOR_REVISION,
            "implementation": "sd14_ddim_epsilon_official_code_loss",
        }
        context["trasce_request"] = request
        context.pop("trasce_applied", None)
        return DefenseDecision(
            allowed=True,
            reason="TraSCE scheduled for generation; no prompt filtering",
            metadata=dict(request),
        )

    def validate_model(self, model_id: str) -> None:
        if model_id != MODEL_ID:
            raise ValueError(
                "TraSCE in this project requires the paper-near Stable Diffusion 1.4 "
                f"backend ({MODEL_ID}); other backends are not supported."
            )

    def generate_image(self, pipeline: Any, *, context: dict[str, Any], **kwargs: Any) -> Any:
        """Called by the Diffusers adapter, never by the evaluation implementation."""
        import torch
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        if not isinstance(pipeline, StableDiffusionPipeline):
            raise TypeError("TraSCE requires the Stable Diffusion 1.4 StableDiffusionPipeline.")
        if not isinstance(pipeline.scheduler, DDIMScheduler):
            raise TypeError("TraSCE requires DDIMScheduler; use configs/models/sd14.yaml.")
        if pipeline.scheduler.config.prediction_type != "epsilon":
            raise ValueError("TraSCE requires epsilon (noise) prediction, not v_prediction.")
        if pipeline.unet.dtype != torch.float32:
            raise ValueError("The paper-near TraSCE preset requires a float32 UNet.")
        request = context.get("trasce_request")
        if not request:
            raise RuntimeError("TraSCE check_prompt must prepare the target before generation.")
        settings = TraSCESettings(**request["parameters"])
        context.pop("trasce_applied", None)
        try:
            with frozen_denoiser(pipeline.unet, settings.gradient_checkpointing):
                result = self._sample(pipeline, request, settings, **kwargs)
            context["trasce_applied"] = True
            return result
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "TraSCE ran out of VRAM with the full method. No reduced guidance, "
                "negative-prompt-only fallback or unprotected image was produced."
            ) from exc
        finally:
            pipeline.maybe_free_model_hooks()

    def _sample(self, pipeline, request, settings, *, prompt, generator, height, width, **_):
        import torch

        device = pipeline._execution_device
        unet = pipeline.unet
        divisor = pipeline.vae_scale_factor
        if height % divisor or width % divisor:
            raise ValueError(f"SD1.4 image dimensions must be divisible by {divisor}.")
        # Native CLIP tokenizer/text encoder, including padding positions.
        conditions = []
        with torch.no_grad():
            for text in ("", prompt, request["negative_prompt"]):
                encoded = pipeline.encode_prompt(
                    prompt=text,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                conditions.append(encoded[0].detach())
                pipeline.maybe_free_model_hooks()
            latents = pipeline.prepare_latents(
                1,
                unet.config.in_channels,
                height,
                width,
                conditions[1].dtype,
                device,
                generator,
                None,
            ).float()
        pipeline.scheduler.set_timesteps(settings.num_inference_steps, device=device)
        diagnostic_rows = []
        with pipeline.progress_bar(total=settings.num_inference_steps) as progress:
            for step, timestep in enumerate(pipeline.scheduler.timesteps, 1):

                def predict(current, branch, step_timestep=timestep):
                    return unet(
                        sample=pipeline.scheduler.scale_model_input(current, step_timestep),
                        timestep=step_timestep,
                        encoder_hidden_states=conditions[branch],
                        return_dict=False,
                    )[0]

                row = {}
                steered, prediction = steer_step(latents, predict, settings, row)
                diagnostic_rows.append(row)
                if step == 1 or step % 5 == 0 or step == settings.num_inference_steps:
                    print(
                        f"TraSCE step {step}/{settings.num_inference_steps}: "
                        + " ".join(
                            f"{key}={value:.6e}"
                            for key, value in row.items()
                            if key != "all_finite"
                        )
                        + f" all_finite={row['all_finite']}",
                        flush=True,
                    )
                with torch.no_grad():
                    latents = pipeline.scheduler.step(
                        prediction, timestep, steered, eta=0.0, return_dict=False
                    )[0].detach()
                progress.update()
        print("TraSCE diagnostics (all denoising steps):", flush=True)
        for key in diagnostic_rows[0]:
            if key == "all_finite":
                continue
            values = [row[key] for row in diagnostic_rows]
            print(
                f"  {key}: min={min(values):.6e} max={max(values):.6e} "
                f"mean={sum(values) / len(values):.6e}",
                flush=True,
            )
        print(f"  all_finite={all(row['all_finite'] for row in diagnostic_rows)}", flush=True)
        with torch.no_grad():
            latents = latents / pipeline.vae.config.scaling_factor
            image = pipeline.vae.decode(latents.to(dtype=pipeline.vae.dtype), return_dict=False)[0]
            image, nsfw = pipeline.run_safety_checker(image, device, conditions[1].dtype)
            images = pipeline.image_processor.postprocess(
                image,
                output_type="pil",
                do_denormalize=[True] if nsfw is None else [not value for value in nsfw],
            )
        return SimpleNamespace(images=images, num_inference_steps=settings.num_inference_steps)

    def check_image(self, image_path: Path, target_concept=None, context=None) -> DefenseDecision:
        if not (context or {}).get("trasce_applied", False):
            raise RuntimeError("TraSCE was not applied by the image model; refusing silent bypass.")
        return DefenseDecision(
            allowed=True,
            reason="TraSCE generation completed; no image classifier or erasure-success claim",
            metadata={"trasce_applied": True, "target_concept": target_concept},
        )
