"""Technical tests for the SDXL SAFREE port; no model download is required."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from t2i_framework.defenses.safree import SAFREEDefense, SAFREESettings
from t2i_framework.defenses.safree_core import (
    LatentReAttention,
    low_frequency_re_attention,
    projection_matrix,
    select_and_project_tokens,
    self_validation_step,
)

torch = pytest.importorskip("torch")


def test_projection_matrix_and_selective_projection() -> None:
    basis = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    projector = projection_matrix(basis)
    assert torch.allclose(projector @ projector, projector)

    token_embeddings = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [2.0, 3.0, 0.0],
            [4.0, 5.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    masked = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    concepts = torch.tensor([[1.0, 0.0, 0.0]])
    result = select_and_project_tokens(token_embeddings, masked, concepts, alpha=0.01)

    assert result.trigger_mask.tolist() == [False, True, False, False]
    assert torch.allclose(result.safe_embeddings[1], torch.tensor([0.0, 3.0, 0.0]))
    assert torch.equal(result.safe_embeddings[2], token_embeddings[2])


def test_self_validation_strength_increases_with_distance() -> None:
    values = [self_validation_step(value) for value in (0.0, 0.5, 1.0)]
    assert values == sorted(values)
    assert values[0] == 0
    assert values[-1] == 10


def test_low_frequency_re_attention_changes_only_safe_branch() -> None:
    features = torch.stack(
        [
            torch.zeros((1, 4, 4)),
            torch.full((1, 4, 4), 2.0),
            torch.ones((1, 4, 4)),
        ]
    )
    filtered = low_frequency_re_attention(features, radius=1, scale=0.5)

    assert torch.equal(filtered[0], features[0])
    assert torch.allclose(filtered[1], torch.ones_like(filtered[1]))
    assert torch.equal(filtered[2], features[2])


def test_re_attention_hooks_are_temporary_and_observable() -> None:
    class UpBlock2D(torch.nn.Module):
        def forward(self, hidden_states, res_hidden_states_tuple):
            return res_hidden_states_tuple[0]

    block = UpBlock2D()
    unet = SimpleNamespace(up_blocks=[block])
    hidden = torch.zeros((3, 2, 4, 4))
    residual = torch.stack(
        [
            torch.zeros((2, 4, 4)),
            torch.full((2, 4, 4), 2.0),
            torch.ones((2, 4, 4)),
        ]
    )
    with LatentReAttention(unet, radius=1, scales={2: 0.5}) as re_attention:
        re_attention.active = True
        filtered = block(hidden, (residual,))
    unfiltered = block(hidden, (residual,))

    assert torch.allclose(filtered[1], torch.ones_like(filtered[1]))
    assert torch.equal(unfiltered, residual)
    assert re_attention.hook_calls == 1
    assert re_attention.filtered_tensors == 1
    assert re_attention.channels_seen == {2}


def test_target_concept_is_the_default_suppression_space() -> None:
    context = {"config": {"defense": {"name": "safree"}}}
    decision = SAFREEDefense().check_prompt(
        "a blue long-eared costume character",
        target_concept="blue rabbit mascot",
        context=context,
    )

    assert decision.allowed
    assert context["safree_request"]["concepts"] == ["blue rabbit mascot"]
    assert context["safree_request"]["negative_prompt"] == "blue rabbit mascot"


def test_configured_concepts_are_cleaned_and_deduplicated() -> None:
    context = {
        "config": {
            "defense": {
                "name": "safree",
                "concept_source": "configured",
                "concepts": [" concept A ", "CONCEPT A", "concept B"],
            }
        }
    }
    SAFREEDefense().check_prompt("prompt", context=context)
    assert context["safree_request"]["concepts"] == ["concept A", "concept B"]


@pytest.mark.parametrize(
    "values",
    [
        {"concept_source": "guess"},
        {"concept_source": "configured", "concepts": "not-a-list"},
        {"alpha": -1},
        {"max_filtered_step": True},
        {"low_frequency_radius": 0},
        {"stage_1280_scale": 1.1},
    ],
)
def test_invalid_settings_fail(values) -> None:
    with pytest.raises((TypeError, ValueError)):
        SAFREESettings(**values)


def test_image_release_requires_all_stages_to_finish() -> None:
    defense = SAFREEDefense()
    with pytest.raises(RuntimeError, match="did not complete"):
        defense.check_image(Path("unused.png"), context={})
    metadata = {"stages_applied": {"latent_re_attention": True}}
    decision = defense.check_image(Path("unused.png"), context={"safree_applied": metadata})
    assert decision.allowed
    assert decision.metadata is metadata


def test_full_sampler_runs_three_branches_and_reports_every_stage(monkeypatch) -> None:
    import t2i_framework.defenses.safree as implementation

    class UpBlock2D(torch.nn.Module):
        def forward(self, hidden_states, res_hidden_states_tuple):
            return res_hidden_states_tuple[0]

    class UNet(torch.nn.Module):
        config = SimpleNamespace(in_channels=1, time_cond_proj_dim=None)

        def __init__(self):
            super().__init__()
            self.up_blocks = torch.nn.ModuleList([UpBlock2D()])

        def forward(self, sample, _timestep, encoder_hidden_states, **_kwargs):
            self.up_blocks[0](sample, (sample.clone(),))
            values = encoder_hidden_states.mean(dim=(1, 2))[:, None, None, None]
            return (sample * 0 + values,)

    class Scheduler:
        order = 1

        def set_timesteps(self, count, device):
            self.timesteps = torch.arange(count - 1, -1, -1, device=device)

        def scale_model_input(self, sample, _timestep):
            return sample

        def step(self, _prediction, _timestep, sample, **_kwargs):
            return (sample,)

    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self):
            pass

    class Tokenizer:
        model_max_length = 5

        def convert_ids_to_tokens(self, values):
            return [f"token-{value}" for value in values]

    class Pipeline:
        _execution_device = torch.device("cpu")
        vae_scale_factor = 1

        def __init__(self):
            self.unet = UNet()
            self.scheduler = Scheduler()
            self.text_encoder = SimpleNamespace(config=SimpleNamespace(hidden_size=1))
            self.text_encoder_2 = SimpleNamespace(config=SimpleNamespace(projection_dim=2))
            self.tokenizer_2 = Tokenizer()
            self._num_timesteps = 0

        def encode_prompt(self, **_kwargs):
            prompt = torch.tensor(
                [
                    [
                        [0.0, 1.0, 0.0],
                        [0.0, 2.0, 1.0],
                        [0.0, 1.0, 2.0],
                        [0.0, 1.0, 1.0],
                        [0.0, 0.5, 0.5],
                    ]
                ]
            )
            negative = torch.zeros_like(prompt)
            pooled = torch.ones((1, 2))
            return prompt, negative, pooled, torch.zeros_like(pooled)

        def prepare_latents(self, *_args):
            return torch.zeros((1, 1, 4, 4))

        def prepare_extra_step_kwargs(self, *_args):
            return {}

        def _get_add_time_ids(self, *_args, **_kwargs):
            return torch.zeros((1, 6))

        def progress_bar(self, total):
            assert total == 2
            return Progress()

    projection = SimpleNamespace(
        safe_embeddings=torch.ones((5, 2)),
        fully_projected_embeddings=torch.ones((5, 2)),
        trigger_mask=torch.tensor([False, True, False, False, False]),
        distances=torch.tensor([0.2, 0.1, 0.1]),
    )
    monkeypatch.setattr(
        implementation,
        "_masked_prompt_embeddings",
        lambda *_args: (
            torch.ones((3, 2)),
            torch.tensor([[1, 2, 3, 4, 5]]),
            torch.ones((1, 5)),
        ),
    )
    monkeypatch.setattr(
        implementation,
        "_concept_embeddings",
        lambda *_args: torch.ones((1, 2)),
    )
    monkeypatch.setattr(implementation, "select_and_project_tokens", lambda *_args: projection)
    monkeypatch.setattr(implementation, "self_validation_step", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        implementation,
        "_decode_sdxl_latents",
        lambda *_args: [SimpleNamespace()],
    )
    request = {
        "prompt": "attacked prompt",
        "concepts": ["target"],
        "negative_prompt": "target",
        "parameters": {
            "concept_source": "target",
            "concepts": [],
            "alpha": 0.01,
            "max_filtered_step": 10,
            "low_frequency_radius": 1,
            "stage_1280_scale": 0.9,
            "stage_640_scale": 0.2,
        },
    }
    # This fake U-Net uses two channels to make the re-attention stage easy to inspect.
    request["parameters"]["stage_1280_scale"] = 0.9
    defense = SAFREEDefense()
    original_context = LatentReAttention
    monkeypatch.setattr(
        implementation,
        "LatentReAttention",
        lambda unet, radius, _scales: original_context(unet, radius, {1: 0.5}),
    )
    result, metadata = defense._sample(
        Pipeline(),
        request,
        prompt="attacked prompt",
        generator=torch.Generator().manual_seed(42),
        num_inference_steps=2,
        guidance_scale=7.5,
        height=4,
        width=4,
    )

    assert len(result.images) == 1
    assert metadata["filtered_steps_used"] == 1
    assert metadata["trigger_tokens"] == ["token-2"]
    assert all(metadata["stages_applied"].values())
    assert metadata["latent_re_attention"]["filtered_tensors"] == 1
