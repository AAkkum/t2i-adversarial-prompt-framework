"""Small mathematical building blocks for the SAFREE generation defense."""

from __future__ import annotations

import math
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any


@dataclass
class ProjectionResult:
    safe_embeddings: Any
    fully_projected_embeddings: Any
    trigger_mask: Any
    distances: Any


def projection_matrix(basis: Any) -> Any:
    """Return the projector onto the columns of a [dimension, vectors] basis."""
    import torch

    if basis.ndim != 2 or basis.shape[1] == 0:
        raise ValueError("Projection basis must contain at least one column vector.")
    basis = basis.float()
    gram = basis.T @ basis
    return basis @ torch.linalg.pinv(gram) @ basis.T


def select_and_project_tokens(
    token_embeddings: Any,
    masked_prompt_embeddings: Any,
    concept_embeddings: Any,
    alpha: float,
) -> ProjectionResult:
    """Detect concept-driving tokens and project only those token embeddings."""
    import torch

    if token_embeddings.ndim != 2:
        raise ValueError("Token embeddings must have shape [tokens, dimension].")
    if masked_prompt_embeddings.ndim != 2 or concept_embeddings.ndim != 2:
        raise ValueError("Masked-prompt and concept embeddings must be two-dimensional.")
    dimension = token_embeddings.shape[1]
    if masked_prompt_embeddings.shape[1] != dimension:
        raise ValueError("Masked-prompt embeddings do not match the token embedding size.")
    if concept_embeddings.shape[1] != dimension:
        raise ValueError("Concept embeddings do not match the token embedding size.")
    if masked_prompt_embeddings.shape[0] > token_embeddings.shape[0] - 2:
        raise ValueError("There are more masked prompts than real prompt-token positions.")

    device = token_embeddings.device
    concept_projector = projection_matrix(concept_embeddings.T).to(device)
    input_projector = projection_matrix(masked_prompt_embeddings.T).to(device)
    identity = torch.eye(dimension, device=device, dtype=torch.float32)

    masked = masked_prompt_embeddings.float()
    residuals = (identity - concept_projector) @ masked.T
    distances = torch.linalg.vector_norm(residuals, dim=0)
    if len(distances) == 1:
        real_token_mask = torch.ones(1, device=device, dtype=torch.bool)
    else:
        leave_one_out_mean = (distances.sum() - distances) / (len(distances) - 1)
        real_token_mask = distances > (1.0 + alpha) * leave_one_out_mean

    # The released SAFREE implementation uses this order. The paper prints
    # PI(I-PC)p; the difference is recorded in docs/SAFREE.md.
    fully_projected = (
        (identity - concept_projector) @ input_projector @ token_embeddings.float().T
    ).T
    sequence_mask = torch.zeros(token_embeddings.shape[0], device=device, dtype=torch.bool)
    sequence_mask[1 : len(real_token_mask) + 1] = real_token_mask
    safe = torch.where(sequence_mask[:, None], fully_projected, token_embeddings.float())
    return ProjectionResult(safe, fully_projected, sequence_mask, distances)


def self_validation_step(
    cosine_distance: float,
    max_step: int = 10,
    midpoint: float = 5.333,
    steepness: float = 2.5,
) -> int:
    """Map prompt/projection distance to the final filtered denoising-step index."""
    scaled = 2.0 * steepness * (10.0 * cosine_distance - midpoint)
    value = 1.0 / (1.0 + math.exp(-scaled))
    return round(max_step * value)


def low_frequency_re_attention(features: Any, radius: int, scale: float) -> Any:
    """Attenuate dominant low-frequency features in the filtered prompt branch."""
    import torch

    if features.ndim != 4 or features.shape[0] != 3:
        raise ValueError("Latent re-attention expects [negative, safe, original] feature branches.")
    if radius < 1 or 2 * radius > min(features.shape[-2:]):
        raise ValueError("The low-frequency radius does not fit the latent feature map.")

    original_dtype = features.dtype
    frequency = torch.fft.fftshift(torch.fft.fftn(features.float(), dim=(-2, -1)), dim=(-2, -1))
    row = features.shape[-2] // 2
    column = features.shape[-1] // 2
    rows = slice(row - radius, row + radius)
    columns = slice(column - radius, column + radius)
    safe_frequency = frequency[1, :, rows, columns]
    original_frequency = frequency[2, :, rows, columns]
    attenuate = safe_frequency.abs() > original_frequency.abs()
    frequency = frequency.clone()
    frequency[1, :, rows, columns] = torch.where(
        attenuate,
        safe_frequency * scale,
        safe_frequency,
    )
    filtered = torch.fft.ifftn(torch.fft.ifftshift(frequency, dim=(-2, -1)), dim=(-2, -1)).real
    return filtered.to(original_dtype)


class LatentReAttention(AbstractContextManager["LatentReAttention"]):
    """Temporarily filter SDXL skip features without replacing Diffusers forwards."""

    def __init__(self, unet: Any, radius: int, scales: dict[int, float]) -> None:
        self.unet = unet
        self.radius = radius
        self.scales = scales
        self.active = False
        self.hook_calls = 0
        self.filtered_tensors = 0
        self.channels_seen: set[int] = set()
        self._handles: list[Any] = []

    def __enter__(self) -> LatentReAttention:
        for block in self.unet.up_blocks:
            if block.__class__.__name__ not in {"UpBlock2D", "CrossAttnUpBlock2D"}:
                continue
            self._handles.append(
                block.register_forward_pre_hook(self._before_up_block, with_kwargs=True)
            )
        if not self._handles:
            raise TypeError("SAFREE found no supported SDXL U-Net upsampling blocks.")
        return self

    def __exit__(self, *_: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _before_up_block(
        self,
        _module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        if not self.active:
            return None
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        residuals = args[1] if len(args) > 1 else kwargs.get("res_hidden_states_tuple")
        if hidden_states is None or residuals is None:
            return None
        channels = int(hidden_states.shape[1])
        scale = self.scales.get(channels)
        if scale is None:
            return None

        self.hook_calls += 1
        self.channels_seen.add(channels)
        filtered_residuals = tuple(
            low_frequency_re_attention(value, self.radius, scale)
            if value.ndim == 4 and value.shape[0] == 3
            else value
            for value in residuals
        )
        self.filtered_tensors += sum(value.ndim == 4 and value.shape[0] == 3 for value in residuals)
        if len(args) > 1:
            changed_args = list(args)
            changed_args[1] = filtered_residuals
            return tuple(changed_args), kwargs
        changed_kwargs = dict(kwargs)
        changed_kwargs["res_hidden_states_tuple"] = filtered_residuals
        return args, changed_kwargs
