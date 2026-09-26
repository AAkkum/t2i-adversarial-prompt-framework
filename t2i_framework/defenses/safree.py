"""Full three-stage SAFREE defense for the SDXL base model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense
from t2i_framework.defenses.safree_core import (
    LatentReAttention,
    select_and_project_tokens,
    self_validation_step,
)

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
AUTHOR_REVISION = "b8b2c3fa9d7f51c46f5a570170503fc98bd9c7ec"


@dataclass
class SAFREESettings:
    concept_source: str = "target"
    concepts: list[str] = field(default_factory=list)
    alpha: float = 0.01
    max_filtered_step: int = 10
    low_frequency_radius: int = 1
    stage_1280_scale: float = 0.9
    stage_640_scale: float = 0.2

    def __post_init__(self) -> None:
        self.concept_source = str(self.concept_source).strip().lower()
        if self.concept_source not in {"target", "configured"}:
            raise ValueError("SAFREE concept_source must be 'target' or 'configured'.")
        if not isinstance(self.concepts, list):
            raise TypeError("SAFREE concepts must be a YAML list of phrases.")
        self.concepts = _clean_phrases(self.concepts)
        _finite_number("alpha", self.alpha, minimum=0.0)
        _integer("max_filtered_step", self.max_filtered_step, minimum=0)
        _integer("low_frequency_radius", self.low_frequency_radius, minimum=1)
        _finite_number("stage_1280_scale", self.stage_1280_scale, minimum=0.0, maximum=1.0)
        _finite_number("stage_640_scale", self.stage_640_scale, minimum=0.0, maximum=1.0)


class SAFREEDefense(Defense):
    """Modify SDXL conditioning and latent features during image generation."""

    name = "safree"

    def check_prompt(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        if context is None:
            raise ValueError("SAFREE requires the shared generation context.")
        settings = _settings_from_context(context)
        if settings.concept_source == "target":
            if not isinstance(target_concept, str) or not target_concept.strip():
                raise ValueError(
                    "SAFREE concept_source=target requires target_concept for every prompt."
                )
            concepts = [target_concept.strip()]
        else:
            concepts = settings.concepts
            if not concepts:
                raise ValueError(
                    "SAFREE concept_source=configured requires at least one concept in YAML."
                )

        request = {
            "prompt": prompt,
            "target_concept": target_concept,
            "concepts": concepts,
            "negative_prompt": ", ".join(concepts),
            "parameters": asdict(settings),
            "author_revision": AUTHOR_REVISION,
            "implementation": "sdxl_full_three_stage_port",
        }
        context["safree_request"] = request
        context.pop("safree_applied", None)
        return DefenseDecision(
            allowed=True,
            reason="SAFREE scheduled for SDXL generation; no prompt was blocked",
            metadata={
                "concept_source": settings.concept_source,
                "concepts": concepts,
                "stages": [
                    "selective_token_projection",
                    "self_validating_filtering",
                    "latent_re_attention",
                ],
            },
        )

    def validate_model(self, model_id: str) -> None:
        if model_id != MODEL_ID:
            raise ValueError(
                "This SAFREE implementation currently supports only "
                f"{MODEL_ID}; received {model_id!r}."
            )

    def generate_image(
        self,
        pipeline: Any,
        *,
        context: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

        if not isinstance(pipeline, StableDiffusionXLPipeline):
            raise TypeError("SAFREE requires a StableDiffusionXLPipeline.")
        if not isinstance(pipeline.scheduler, DPMSolverMultistepScheduler):
            raise TypeError(
                "SAFREE requires DPMSolverMultistepScheduler; use the SDXL SAFREE model preset."
            )
        request = context.get("safree_request")
        if not request:
            raise RuntimeError("SAFREE check_prompt must prepare the concepts before generation.")
        if kwargs.get("prompt") != request["prompt"]:
            raise RuntimeError(
                "SAFREE received a different prompt than the prepared defense request."
            )

        context.pop("safree_applied", None)
        try:
            with torch.inference_mode():
                result, metadata = self._sample(pipeline, request, **kwargs)
            context["safree_applied"] = metadata
            return result
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "SAFREE ran out of VRAM. No unprotected fallback image was generated."
            ) from exc
        finally:
            pipeline.maybe_free_model_hooks()

    def check_image(
        self,
        image_path: Path,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        metadata = (context or {}).get("safree_applied")
        if not metadata:
            raise RuntimeError(
                "SAFREE generation did not complete, so the image cannot be released."
            )
        return DefenseDecision(
            allowed=True,
            reason="SAFREE completed all three generation-time defense stages",
            metadata=metadata,
        )

    def _sample(
        self,
        pipeline: Any,
        request: dict[str, Any],
        *,
        prompt: str,
        generator: Any,
        num_inference_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        **_: Any,
    ) -> tuple[Any, dict[str, Any]]:
        import torch
        import torch.nn.functional as functional
        from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
            StableDiffusionXLPipelineOutput,
        )

        settings = SAFREESettings(**request["parameters"])
        if guidance_scale <= 1.0:
            raise ValueError("SAFREE requires classifier-free guidance greater than 1.0.")
        if pipeline.unet.config.time_cond_proj_dim is not None:
            raise ValueError("SAFREE does not support SDXL models with timestep conditioning.")
        if width % pipeline.vae_scale_factor or height % pipeline.vae_scale_factor:
            raise ValueError(f"SDXL dimensions must be divisible by {pipeline.vae_scale_factor}.")

        device = pipeline._execution_device
        pipeline._guidance_scale = guidance_scale
        pipeline._guidance_rescale = 0.0
        pipeline._clip_skip = None
        pipeline._cross_attention_kwargs = None
        pipeline._denoising_end = None
        pipeline._interrupt = False

        (
            prompt_embeddings,
            negative_embeddings,
            pooled_prompt,
            negative_pooled_prompt,
        ) = pipeline.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=request["negative_prompt"],
        )
        first_encoder_size = int(pipeline.text_encoder.config.hidden_size)
        original_second = prompt_embeddings[0, :, first_encoder_size:]
        masked_embeddings, token_ids, attention_mask = _masked_prompt_embeddings(
            pipeline, prompt, device
        )
        concept_embeddings = _concept_embeddings(pipeline, request["concepts"], device)
        projection = select_and_project_tokens(
            original_second,
            masked_embeddings,
            concept_embeddings,
            settings.alpha,
        )

        safe_prompt_embeddings = prompt_embeddings.clone()
        safe_prompt_embeddings[0, :, first_encoder_size:] = projection.safe_embeddings.to(
            prompt_embeddings.dtype
        )
        active_positions = attention_mask[0].bool()
        cosine = functional.cosine_similarity(
            original_second[active_positions].float(),
            projection.fully_projected_embeddings[active_positions].float(),
            dim=-1,
        ).mean()
        cosine_distance = float((1.0 - cosine).item())
        final_filtered_step = self_validation_step(
            cosine_distance,
            max_step=settings.max_filtered_step,
        )

        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipeline.scheduler.timesteps
        latents = pipeline.prepare_latents(
            1,
            pipeline.unet.config.in_channels,
            height,
            width,
            prompt_embeddings.dtype,
            device,
            generator,
            None,
        )
        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, 0.0)
        add_time_ids = pipeline._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=prompt_embeddings.dtype,
            text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim,
        ).to(device)
        negative_time_ids = add_time_ids.clone()
        branch_pooled = torch.cat([negative_pooled_prompt, pooled_prompt, pooled_prompt], dim=0).to(
            device
        )
        branch_time_ids = torch.cat([negative_time_ids, add_time_ids, add_time_ids], dim=0).to(
            device
        )

        filtered_steps_used = 0
        scales = {
            1280: settings.stage_1280_scale,
            640: settings.stage_640_scale,
        }
        pipeline._num_timesteps = len(timesteps)
        with LatentReAttention(
            pipeline.unet,
            settings.low_frequency_radius,
            scales,
        ) as re_attention:
            with pipeline.progress_bar(total=num_inference_steps) as progress:
                for step_index, timestep in enumerate(timesteps):
                    filtering_active = step_index <= final_filtered_step
                    if filtering_active:
                        filtered_steps_used += 1
                    re_attention.active = filtering_active
                    current_prompt = (
                        safe_prompt_embeddings if filtering_active else prompt_embeddings
                    )
                    branch_embeddings = torch.cat(
                        [negative_embeddings, current_prompt, prompt_embeddings], dim=0
                    ).to(device)
                    latent_input = pipeline.scheduler.scale_model_input(
                        torch.cat([latents] * 3), timestep
                    )
                    prediction = pipeline.unet(
                        latent_input,
                        timestep,
                        encoder_hidden_states=branch_embeddings,
                        added_cond_kwargs={
                            "text_embeds": branch_pooled,
                            "time_ids": branch_time_ids,
                        },
                        return_dict=False,
                    )[0]
                    negative_prediction, safe_prediction, _original_prediction = prediction.chunk(3)
                    guided_prediction = negative_prediction + guidance_scale * (
                        safe_prediction - negative_prediction
                    )
                    latents = pipeline.scheduler.step(
                        guided_prediction,
                        timestep,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0]
                    progress.update()

        image = _decode_sdxl_latents(pipeline, latents)
        trigger_positions = projection.trigger_mask.nonzero(as_tuple=False).flatten().tolist()
        tokens = pipeline.tokenizer_2.convert_ids_to_tokens(token_ids[0].tolist())
        metadata = {
            "implementation": "sdxl_full_three_stage_port",
            "author_revision": AUTHOR_REVISION,
            "concept_source": settings.concept_source,
            "concepts": request["concepts"],
            "alpha": settings.alpha,
            "trigger_token_positions": trigger_positions,
            "trigger_tokens": [tokens[position] for position in trigger_positions],
            "masked_token_distances": [float(value) for value in projection.distances.tolist()],
            "self_validation_cosine_distance": cosine_distance,
            "self_validation_final_step": final_filtered_step,
            "filtered_steps_used": filtered_steps_used,
            "num_inference_steps": num_inference_steps,
            "latent_re_attention": {
                "enabled": True,
                "radius": settings.low_frequency_radius,
                "scales": {str(key): value for key, value in scales.items()},
                "hook_calls": re_attention.hook_calls,
                "filtered_tensors": re_attention.filtered_tensors,
                "channels_seen": sorted(re_attention.channels_seen),
            },
            "stages_applied": {
                "selective_token_projection": True,
                "self_validating_filtering": True,
                "latent_re_attention": re_attention.filtered_tensors > 0,
            },
        }
        if not metadata["stages_applied"]["latent_re_attention"]:
            raise RuntimeError(
                "SAFREE latent re-attention did not reach a supported SDXL U-Net stage."
            )
        return StableDiffusionXLPipelineOutput(images=image), metadata


def _settings_from_context(context: dict[str, Any]) -> SAFREESettings:
    config = dict((context.get("config") or {}).get("defense", {}))
    config.pop("name", None)
    return SAFREESettings(**config)


def _masked_prompt_embeddings(
    pipeline: Any,
    prompt: str,
    device: Any,
) -> tuple[Any, Any, Any]:
    tokenizer = pipeline.tokenizer_2
    shortest = tokenizer(prompt, padding="longest", return_tensors="pt")
    token_ids = shortest.input_ids[:, : tokenizer.model_max_length]
    real_tokens = min(shortest.input_ids.shape[1] - 2, tokenizer.model_max_length - 2)
    if real_tokens < 1:
        raise ValueError("SAFREE requires at least one real prompt token.")
    masked_ids = token_ids.repeat(real_tokens, 1)
    for index in range(real_tokens):
        masked_ids[index, index + 1] = 0
    masked_outputs = pipeline.text_encoder_2(masked_ids.to(device))
    masked_embeddings = _pooled_embeddings(masked_outputs)
    padded = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return masked_embeddings, padded.input_ids, padded.attention_mask.to(device)


def _concept_embeddings(pipeline: Any, concepts: list[str], device: Any) -> Any:
    encoded = pipeline.tokenizer_2(
        concepts,
        padding="max_length",
        max_length=pipeline.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return _pooled_embeddings(pipeline.text_encoder_2(encoded.input_ids.to(device)))


def _pooled_embeddings(outputs: Any) -> Any:
    pooled = getattr(outputs, "text_embeds", None)
    return outputs[0] if pooled is None else pooled


def _decode_sdxl_latents(pipeline: Any, latents: Any) -> list[Any]:
    import torch

    needs_upcasting = pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast
    if needs_upcasting:
        pipeline.upcast_vae()
        latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)
    latents = latents / pipeline.vae.config.scaling_factor
    image = pipeline.vae.decode(latents, return_dict=False)[0]
    if needs_upcasting:
        pipeline.vae.to(dtype=torch.float16)
    if pipeline.watermark is not None:
        image = pipeline.watermark.apply_watermark(image)
    return pipeline.image_processor.postprocess(image, output_type="pil")


def _clean_phrases(values: list[Any]) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for value in values:
        phrase = str(value).strip()
        key = phrase.casefold()
        if phrase and key not in seen:
            phrases.append(phrase)
            seen.add(key)
    return phrases


def _finite_number(
    name: str,
    value: Any,
    *,
    minimum: float,
    maximum: float | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError(f"SAFREE {name} must be a finite number.")
    if value < minimum or (maximum is not None and value > maximum):
        limit = f" between {minimum} and {maximum}" if maximum is not None else f" >= {minimum}"
        raise ValueError(f"SAFREE {name} must be{limit}.")


def _integer(name: str, value: Any, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"SAFREE {name} must be an integer >= {minimum}.")
