"""TECHNICAL SMOKE TEST / NOT PAPER-COMPARABLE: no downloaded models or ASR."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from main import app
from t2i_framework.core.config import load_yaml_config
from t2i_framework.core.registry import build_attack, build_defense
from t2i_framework.defenses.trasce import (
    MODEL_ID,
    PRESETS,
    TraSCEDefense,
    TraSCESettings,
    frozen_denoiser,
    localized_loss,
    steer_step,
)
from t2i_framework.evaluation.runner import ExperimentRunner
from t2i_framework.models.diffusers_model import DiffusersImageModel

torch = pytest.importorskip("torch")


def test_registry_and_published_object_preset():
    assert isinstance(build_defense("trasce"), TraSCEDefense)
    config = load_yaml_config(Path("configs/defenses/trasce.yaml"))["defense"]
    config.pop("name")
    settings = TraSCESettings(**config)
    assert settings == TraSCESettings()
    assert (
        settings.guidance_loss_scale,
        settings.discriminator_guidance_scale,
        settings.sigma,
    ) == (1, 1, 1)
    assert (settings.num_inference_steps, settings.guidance_scale) == (50, 7.5)


@pytest.mark.parametrize("category, expected", list(PRESETS.items()))
def test_category_parameters(category, expected):
    settings = TraSCESettings(category=category)
    assert (
        settings.guidance_loss_scale,
        settings.discriminator_guidance_scale,
        settings.sigma,
    ) == expected


@pytest.mark.parametrize(
    "config",
    [
        {"category": "guess"},
        {"sigma": 0},
        {"sigma": float("nan")},
        {"guidance_loss_scale": 0},
        {"discriminator_guidance_scale": -1},
        {"guidance_scale": True},
        {"num_inference_steps": True},
        {"gradient_checkpointing": "yes"},
        {"negative_prompt": ""},
    ],
)
def test_bad_configuration_fails(config):
    with pytest.raises((ValueError, TypeError)):
        TraSCESettings(**config)


def test_target_and_adversarial_prompt_are_not_rewritten():
    defense = TraSCEDefense()
    context = {"config": {"defense": {"name": "trasce"}}}
    decision = defense.check_prompt("unrelated decoded adversarial words", "DoG", context)
    assert decision.allowed
    assert decision.metadata["target_concept"] == "DoG"
    assert context["trasce_request"]["negative_prompt"] == "DoG"
    assert context["trasce_request"]["parameters"]["category"] == "object"
    with pytest.raises(RuntimeError, match="not applied"):
        defense.check_image(Path("unused.png"), "DoG", context)


def test_explicit_descriptions_and_category_do_not_relabel_target():
    context = {
        "config": {
            "defense": {
                "category": "style",
                "concept_erasure": "artist style",
                "negative_prompt": "specified style",
            }
        }
    }
    decision = TraSCEDefense().check_prompt("image prompt", "original target", context)
    assert decision.metadata["target_concept"] == "original target"
    assert decision.metadata["concept_erasure"] == "artist style"
    assert decision.metadata["negative_prompt"] == "specified style"
    assert decision.metadata["parameters"]["sigma"] == 0.25


def test_loss_is_unsquared_global_l2_from_official_code():
    settings = TraSCESettings(guidance_loss_scale=1.5, sigma=2)
    p = torch.tensor([3.0, 4.0])
    n = torch.zeros(2)
    assert localized_loss(p, n, settings).item() == pytest.approx(
        -1.5 * torch.exp(torch.tensor(-2.5)).item()
    )


def test_latent_gradient_uses_both_branches_and_stale_predictions():
    x = torch.tensor([0.1, -0.3])
    settings = TraSCESettings(guidance_loss_scale=1.5, discriminator_guidance_scale=2, sigma=4)
    calls = []

    def predict(current, branch):
        calls.append((branch, torch.is_grad_enabled()))
        return (0.5 * current, 2 * current + 3, -current + 1)[branch]

    steered, velocity = steer_step(x, predict, settings)
    delta = 3 * x + 2
    distance = delta.norm()
    gradient = (1.5 / 4) * torch.exp(-distance / 4) * delta / distance * 3
    assert torch.allclose(steered, x - 2 * gradient)
    assert torch.allclose(velocity, 0.5 * x + 7.5 * delta)
    assert calls == [(0, False), (1, True), (2, True)]
    assert not steered.requires_grad and not velocity.requires_grad
    assert x.grad is None


def test_equal_prompt_and_negative_reduces_to_unconditional():
    x = torch.tensor([0.2, 0.4])
    steered, velocity = steer_step(x, lambda z, b: z * (2 if b else 3), TraSCESettings())
    assert torch.equal(steered, x)
    assert torch.equal(velocity, 3 * x)


def test_diagnostics_are_observational_and_measure_actual_update():
    x = torch.tensor([0.1, -0.3])
    settings = TraSCESettings()

    def predict(z, branch):
        return z * (branch + 1) + branch

    expected = steer_step(x, predict, settings)
    row = {}
    actual = steer_step(x, predict, settings, row)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert row["steering_norm"] == pytest.approx((actual[0] - x).norm().item())
    assert row["relative_steering"] == pytest.approx(row["steering_norm"] / x.norm().item())
    assert row["all_finite"]
    assert all(not isinstance(value, torch.Tensor) for value in row.values())


def test_diagnostics_expose_underflow_and_undefined_zero_latent_ratio():
    row = {}
    steer_step(torch.ones(2), lambda z, b: z + b * 1000, TraSCESettings(), row)
    assert row["loss"] == 0 and row["gradient_norm"] == 0
    assert row["steering_norm"] == 0 and row["all_finite"]
    steer_step(torch.zeros(2), lambda z, b: z, TraSCESettings(), row)
    assert math.isnan(row["relative_steering"])
    assert not row["all_finite"]


def test_initial_latents_match_normal_sd14(monkeypatch, capsys):
    from diffusers import StableDiffusionPipeline

    prepare = StableDiffusionPipeline.prepare_latents
    pipe = fake_pipeline(monkeypatch)
    encode = pipe.encode_prompt
    pipe.encode_prompt = lambda **kw: tuple(
        v.to(torch.float32) if v is not None else None for v in encode(**kw)
    )
    initial = []

    def capture(*args):
        result = prepare(pipe, *args)
        initial.append(result.clone())
        return result

    pipe.prepare_latents = capture
    expected = prepare(
        pipe,
        1,
        2,
        4,
        4,
        torch.float32,
        torch.device("cpu"),
        torch.Generator().manual_seed(42),
        None,
    )
    defense = TraSCEDefense()
    context = {"config": {"defense": {"num_inference_steps": 2}}}
    defense.check_prompt("adversarial", "dog", context)
    defense.generate_image(
        pipe,
        context=context,
        prompt="adversarial",
        generator=torch.Generator().manual_seed(42),
        height=4,
        width=4,
    )
    assert initial[0].dtype == torch.float32
    assert torch.equal(initial[0], expected)
    output = capsys.readouterr().out
    assert "TraSCE step 1/2" in output and "TraSCE step 2/2" in output
    assert "TraSCE diagnostics" in output and "all_finite=True" in output


def test_no_grad_outer_scope_is_supported_but_inference_mode_rejected():
    with torch.no_grad():
        steer_step(torch.ones(1), lambda z, b: z * (b + 1), TraSCESettings())
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        steer_step(torch.ones(1), lambda z, b: z * (b + 1), TraSCESettings())


def test_nonfinite_prediction_fails_without_fallback():
    with pytest.raises(FloatingPointError):
        steer_step(torch.ones(1), lambda z, b: z * float("nan"), TraSCESettings())


def tiny_unet():
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel(
        sample_size=4,
        in_channels=2,
        out_channels=2,
        down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",),
        block_out_channels=(8,),
        layers_per_block=1,
        cross_attention_dim=8,
        attention_head_dim=2,
        norm_num_groups=4,
    )


def test_real_sd14_unet_gradients_checkpoint_parity_and_immutable_weights():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        unet = tiny_unet()
        x = torch.randn(1, 2, 4, 4)
        conditions = [torch.randn(1, 3, 8) for _ in range(3)]
    before = {k: v.clone() for k, v in unet.state_dict().items()}
    flags = [p.requires_grad for p in unet.parameters()]
    training = unet.training
    results = []
    for checkpointing in (False, True):
        with frozen_denoiser(unet, checkpointing):

            def predict(z, branch):
                return unet(
                    sample=z,
                    encoder_hidden_states=conditions[branch],
                    timestep=torch.tensor([500.0]),
                    return_dict=False,
                )[0]

            results.append(steer_step(x, predict, TraSCESettings()))
    # Compare against the author's batched three-condition loss/update order.
    with frozen_denoiser(unet, False), torch.enable_grad():
        original = x.detach().requires_grad_(True)
        u, p, n = unet(
            sample=torch.cat([original] * 3),
            encoder_hidden_states=torch.cat(conditions),
            timestep=torch.tensor([500.0]),
            return_dict=False,
        )[0].chunk(3)
        gradient = torch.autograd.grad(localized_loss(p, n, TraSCESettings()), original)[0]
        batched = (original.detach() - gradient, u.detach() + 7.5 * (p.detach() - n.detach()))
    assert all(torch.allclose(a, b, atol=2e-5) for a, b in zip(results[0], batched))
    assert not torch.equal(results[0][0], x)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(*results))
    assert all(torch.equal(before[k], v) for k, v in unet.state_dict().items())
    assert all(p.grad is None for p in unet.parameters())
    assert [p.requires_grad for p in unet.parameters()] == flags
    assert unet.training == training and not unet.is_gradient_checkpointing
    with pytest.raises(RuntimeError, match="injected"), frozen_denoiser(unet, True):
        raise RuntimeError("injected")
    assert [p.requires_grad for p in unet.parameters()] == flags
    assert unet.training == training and not unet.is_gradient_checkpointing


class Progress:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self):
        pass


def fake_pipeline(monkeypatch):
    import diffusers
    from PIL import Image

    class UNet(torch.nn.Module):
        config = SimpleNamespace(in_channels=2, patch_size=1)
        dtype = torch.float32
        is_gradient_checkpointing = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))

        def enable_gradient_checkpointing(self):
            self.is_gradient_checkpointing = True

        def disable_gradient_checkpointing(self):
            self.is_gradient_checkpointing = False

        def forward(self, sample, encoder_hidden_states, **kwargs):
            scale = encoder_hidden_states.mean()
            return (sample * (self.weight + scale),)

    class VAE:
        config = SimpleNamespace(scaling_factor=2.0, shift_factor=0.5)
        dtype = torch.float32

        def decode(self, x, **kwargs):
            return (x,)

    class Pipeline:
        _execution_device = torch.device("cpu")
        vae_scale_factor = 1

        def __init__(self):
            self.unet = UNet()
            self.scheduler = diffusers.DDIMScheduler(clip_sample=False, steps_offset=1)
            self.vae = VAE()
            self.encodings = []
            self.frees = 0
            self.decoded = None
            self.image_processor = SimpleNamespace(postprocess=self.postprocess)

        def postprocess(self, x, output_type, **kwargs):
            self.decoded = x.clone()
            return [Image.new("RGB", (8, 8), "white")]

        def encode_prompt(self, prompt, **kwargs):
            self.encodings.append(prompt)
            assert not torch.is_grad_enabled()
            value = torch.tensor([len(prompt) / 100])
            return value, None

        def run_safety_checker(self, image, device, dtype):
            return image, None

        def prepare_latents(
            self, batch, channels, height, width, dtype, device, generator, latents
        ):
            return torch.randn((batch, channels, height, width), generator=generator, dtype=dtype)

        def progress_bar(self, total):
            return Progress()

        def maybe_free_model_hooks(self):
            self.frees += 1

        def __call__(self, **kwargs):
            raise AssertionError("TraSCE must not call ordinary negative-prompt generation")

    monkeypatch.setattr(diffusers, "StableDiffusionPipeline", Pipeline)
    return Pipeline()


def test_sampling_deterministic_target_and_model_reuse(monkeypatch):
    pipe = fake_pipeline(monkeypatch)
    defense = TraSCEDefense()
    context = {"config": {"defense": {"num_inference_steps": 2}}}
    before = pipe.unet.weight.detach().clone()
    outputs = []
    for _ in range(2):
        defense.check_prompt("adversarial text", "dog", context)
        defense.generate_image(
            pipe,
            context=context,
            prompt="adversarial text",
            generator=torch.Generator().manual_seed(42),
            height=4,
            width=4,
        )
        outputs.append(pipe.decoded)
        assert defense.check_image(Path("unused"), "dog", context).allowed
    assert torch.equal(*outputs)
    assert pipe.encodings == ["", "adversarial text", "dog"] * 2
    assert torch.equal(before, pipe.unet.weight)
    assert pipe.unet.weight.grad is None
    assert pipe.frees == 8


def test_generation_failure_restores_state_and_never_marks_applied(monkeypatch):
    pipe = fake_pipeline(monkeypatch)
    defense = TraSCEDefense()
    context = {}
    defense.check_prompt("a dog", "dog", context)
    initial = pipe.unet.weight.clone()

    def oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("injected")

    monkeypatch.setattr(defense, "_sample", oom)
    with pytest.raises(RuntimeError, match="No reduced guidance"):
        defense.generate_image(pipe, context=context)
    assert pipe.frees == 1
    assert pipe.unet.weight.requires_grad
    assert not pipe.unet.is_gradient_checkpointing
    assert torch.equal(initial, pipe.unet.weight)
    assert not context.get("trasce_applied")
    with pytest.raises(RuntimeError, match="not applied"):
        defense.check_image(Path("unused"), "dog", context)


def test_scheduler_receives_exact_steered_latent(monkeypatch):
    import t2i_framework.defenses.trasce as implementation

    pipe = fake_pipeline(monkeypatch)
    expected = []
    original_step = pipe.scheduler.step

    def capture_steering(*args):
        result = steer_step(*args)
        expected.append(result)
        return result

    def verify_step(prediction, timestep, sample, **kwargs):
        steered, velocity = expected[-1]
        assert sample is steered and prediction is velocity
        return original_step(prediction, timestep, sample, **kwargs)

    monkeypatch.setattr(implementation, "steer_step", capture_steering)
    monkeypatch.setattr(pipe.scheduler, "step", verify_step)
    defense = TraSCEDefense()
    context = {"config": {"defense": {"num_inference_steps": 2}}}
    defense.check_prompt("adversarial", "dog", context)
    defense.generate_image(
        pipe,
        context=context,
        prompt="adversarial",
        generator=torch.Generator().manual_seed(42),
        height=4,
        width=4,
    )
    assert len(expected) == 2


def test_unsupported_scheduler_and_missing_target_fail(monkeypatch):
    pipe = fake_pipeline(monkeypatch)
    defense = TraSCEDefense()
    with pytest.raises(ValueError, match="target_concept"):
        defense.check_prompt("a dog", None, {})
    context = {}
    defense.check_prompt("a dog", "dog", context)
    pipe.scheduler = object()
    with pytest.raises(TypeError, match="DDIMScheduler"):
        defense.generate_image(pipe, context=context)


def test_velocity_prediction_and_low_precision_are_rejected(monkeypatch):
    from diffusers import DDIMScheduler

    pipe = fake_pipeline(monkeypatch)
    pipe.scheduler = DDIMScheduler(prediction_type="v_prediction")
    with pytest.raises(ValueError, match="epsilon"):
        TraSCEDefense().generate_image(pipe, context={})
    pipe.scheduler = DDIMScheduler()
    pipe.unet.dtype = torch.float16
    with pytest.raises(ValueError, match="float32"):
        TraSCEDefense().generate_image(pipe, context={})


def test_sd14_config_matches_author_sampling_defaults():
    config = load_yaml_config(Path("configs/models/sd14.yaml"))["model"]
    assert config["model_id"] == MODEL_ID
    assert config["scheduler"] == "ddim"
    assert config["dtype"] == "float32"
    assert config["disable_safety_checker"] is True
    assert config["width"] == config["height"] == 512
    assert config["num_inference_steps"] == TraSCESettings().num_inference_steps == 50
    assert config["guidance_scale"] == TraSCESettings().guidance_scale == 7.5


def test_sd35_none_still_uses_native_pipeline(monkeypatch, tmp_path):
    import diffusers
    from PIL import Image

    loads, calls = [], []
    native_scheduler = object()

    class Pipeline:
        scheduler = native_scheduler

        def enable_model_cpu_offload(self, device):
            assert device == "cuda"

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (8, 8))])

    pipe = Pipeline()

    def load(model_id, **kwargs):
        loads.append((model_id, kwargs))
        return pipe

    monkeypatch.setattr(diffusers.AutoPipelineForText2Image, "from_pretrained", load)
    config = load_yaml_config(Path("configs/models/sd35_medium.yaml"))["model"]
    config.pop("name")
    config.update(device="cuda", enable_model_cpu_offload=True)
    result = DiffusersImageModel(**config).generate(
        "a dog", tmp_path, 42, {"defense": build_defense("none")}
    )
    assert result.image_path.is_file()
    assert pipe.scheduler is native_scheduler
    assert loads == [
        (
            "stabilityai/stable-diffusion-3.5-medium",
            {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True},
        )
    ]
    assert calls[0]["generator"].initial_seed() == 42
    assert calls[0]["num_inference_steps"] == 20
    assert calls[0]["guidance_scale"] == 4.5


def test_sd14_adapter_sets_ddim_and_classifier_choice_for_both_defenses(monkeypatch, tmp_path):
    import diffusers
    from PIL import Image

    loads = []

    class Pipeline:
        def __init__(self):
            self.scheduler = diffusers.PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                steps_offset=1,
                skip_prk_steps=True,
            )

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            assert isinstance(self.scheduler, diffusers.DDIMScheduler)
            assert self.scheduler.config.prediction_type == "epsilon"
            assert kwargs["num_inference_steps"] == 50 and kwargs["guidance_scale"] == 7.5
            return SimpleNamespace(images=[Image.new("RGB", (8, 8))])

    def load(*args, **kwargs):
        loads.append(kwargs)
        return Pipeline()

    monkeypatch.setattr(diffusers.AutoPipelineForText2Image, "from_pretrained", load)
    config = load_yaml_config(Path("configs/models/sd14.yaml"))["model"]
    config.pop("name")
    config.update(device="cpu", enable_model_cpu_offload=False)
    for name in ("none", "trasce"):
        defense = build_defense(name)
        if name == "trasce":
            monkeypatch.setattr(defense, "generate_image", lambda pipe, context, **kw: pipe(**kw))
        DiffusersImageModel(**config).generate("dog", tmp_path, 42, {"defense": defense})
    assert (
        loads[0]
        == loads[1]
        == {
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
            "safety_checker": None,
            "requires_safety_checker": False,
        }
    )


def test_actual_native_sd_pipeline_and_trasce_share_initial_latents_and_schedule(monkeypatch):
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline

    with torch.random.fork_rng():
        torch.manual_seed(42)
        pipe = StableDiffusionPipeline(
            vae=AutoencoderKL(
                block_out_channels=(8,),
                norm_num_groups=4,
                latent_channels=2,
                sample_size=8,
            ),
            unet=tiny_unet(),
            text_encoder=None,
            tokenizer=None,
            scheduler=DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
            ),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        embeddings = {text: torch.randn(1, 3, 8) for text in ("", "adversarial", "dog")}

    def encode(prompt, *args, **kwargs):
        return embeddings[prompt], embeddings[""]

    monkeypatch.setattr(pipe, "encode_prompt", encode)
    initial, traces = [], []
    prepare, step = pipe.prepare_latents, pipe.scheduler.step

    def capture_initial(*args, **kwargs):
        value = prepare(*args, **kwargs)
        initial.append(value.clone())
        return value

    def capture_step(prediction, timestep, sample, **kwargs):
        traces.append((int(timestep), sample.dtype, prediction.dtype, kwargs.get("eta", 0.0)))
        return step(prediction, timestep, sample, **kwargs)

    monkeypatch.setattr(pipe, "prepare_latents", capture_initial)
    monkeypatch.setattr(pipe.scheduler, "step", capture_step)
    native = pipe(
        prompt="adversarial",
        height=8,
        width=8,
        num_inference_steps=2,
        guidance_scale=7.5,
        generator=torch.Generator().manual_seed(42),
    )
    defense = TraSCEDefense()
    context = {"config": {"defense": {"num_inference_steps": 2}}}
    defense.check_prompt("adversarial", "dog", context)
    defended = defense.generate_image(
        pipe,
        context=context,
        prompt="adversarial",
        height=8,
        width=8,
        generator=torch.Generator().manual_seed(42),
    )
    assert len(initial) == 2 and torch.equal(initial[0], initial[1])
    assert traces[:2] == traces[2:]
    assert native.images[0].size == defended.images[0].size == (8, 8)
    assert context["trasce_applied"]


def test_adapter_hook_and_original_runner_with_identity(monkeypatch, tmp_path):
    pipe = fake_pipeline(monkeypatch)
    model = DiffusersImageModel(model_id=MODEL_ID, device="cpu", dtype="float32", width=4, height=4)
    model._pipeline = pipe
    result = ExperimentRunner(
        model, build_attack("identity"), build_defense("trasce"), tmp_path / "out"
    ).run([("a dog sitting in a park", "DoG")], 42, {"defense": {"num_inference_steps": 2}})[0]
    assert result.original_prompt == result.attacked_prompt == "a dog sitting in a park"
    assert result.target_concept == "DoG"
    assert pipe.encodings == ["", result.attacked_prompt, "DoG"]
    assert result.metadata["image_defense"]["metadata"]["trasce_applied"]
    assert Path(result.generated_image_path).is_file()


@pytest.mark.parametrize("model_id", ["not-sd14", "stabilityai/stable-diffusion-3.5-medium"])
def test_invalid_model_rejected_before_loading(monkeypatch, tmp_path, model_id):
    import diffusers

    def forbidden(*args, **kwargs):
        raise AssertionError("must not download incompatible model")

    monkeypatch.setattr(diffusers.AutoPipelineForText2Image, "from_pretrained", forbidden)
    with pytest.raises(ValueError, match="paper-near Stable Diffusion 1.4"):
        DiffusersImageModel(model_id=model_id, device="cpu").generate(
            "a dog", tmp_path, 42, {"defense": TraSCEDefense()}
        )


def test_cli_rejects_sd35_trasce_before_attack_search(monkeypatch, tmp_path):
    from t2i_framework.attacks.ring_a_bell import RingABellAttack

    def forbidden(*args, **kwargs):
        raise AssertionError("incompatible backend must fail before expensive attack search")

    monkeypatch.setattr(RingABellAttack, "generate", forbidden)
    result = CliRunner().invoke(
        app,
        [
            "--model",
            "diffusers",
            "--model-config",
            "configs/models/sd35_medium.yaml",
            "--attack",
            "ring_a_bell",
            "--defense",
            "trasce",
            "--prompt",
            "a dog",
            "--target",
            "dog",
            "--out",
            str(tmp_path / "unused"),
        ],
    )
    assert result.exit_code == 2
    # Rich may wrap the backend name across console lines.
    assert "paper-near" in result.output and "1.4" in result.output
    assert not (tmp_path / "unused").exists()


def test_model_offload_hook_keeps_cpu_generator_and_effective_step_count(monkeypatch, tmp_path):
    import diffusers

    pipe = fake_pipeline(monkeypatch)
    offload_calls = []
    pipe.enable_model_cpu_offload = lambda device: offload_calls.append(device)
    monkeypatch.setattr(
        diffusers.AutoPipelineForText2Image, "from_pretrained", lambda *a, **k: pipe
    )
    model = DiffusersImageModel(
        model_id=MODEL_ID,
        device="cuda",
        dtype="bfloat16",
        width=4,
        height=4,
        enable_model_cpu_offload=True,
    )
    defense = TraSCEDefense()
    context = {"defense": defense, "config": {"defense": {"num_inference_steps": 2}}}
    defense.check_prompt("a dog", "dog", context)
    result = model.generate("a dog", tmp_path, 42, context)
    assert offload_calls == ["cuda"]
    assert result.metadata["num_inference_steps"] == 2
    assert result.metadata["model_cpu_offload"]
    assert context["trasce_applied"]


@pytest.mark.parametrize("target_visible", [True, False])
def test_cli_configuration_and_adversarial_text_reach_trasce(monkeypatch, tmp_path, target_visible):
    import diffusers

    from t2i_framework.attacks.ring_a_bell import RingABellAttack
    from t2i_framework.core.types import AttackCandidate

    pipe = fake_pipeline(monkeypatch)
    pipe.to = lambda device: pipe
    monkeypatch.setattr(
        diffusers.AutoPipelineForText2Image, "from_pretrained", lambda *a, **k: pipe
    )

    def attack(self, prompt, target_concept, context):
        assert prompt == "a dog sitting in a park" and target_concept == "dog"
        return [AttackCandidate(text="adversarial tokens", metadata={})]

    monkeypatch.setattr(RingABellAttack, "generate", attack)
    from t2i_framework.core.local_model_client import LocalMultimodalClient

    judge_calls = []

    def judge(self, system_prompt, user_prompt, image_path=None, **kwargs):
        assert image_path.is_file()
        assert "a dog sitting in a park" in user_prompt
        assert "adversarial tokens" in user_prompt
        assert "Target concept:\ndog" in user_prompt
        judge_calls.append(image_path)
        return json.dumps(
            {
                "target_visible": target_visible,
                "intent_preserved": True,
                "policy_violation": False,
                "confidence": 0.99,
                "categories": [],
                "reason": "Mocked image judgment for integration test",
            }
        )

    monkeypatch.setattr(LocalMultimodalClient, "complete", judge)
    config = tmp_path / "test.json"
    config.write_text(
        json.dumps(
            {
                "model": {
                    "name": "diffusers",
                    "model_id": MODEL_ID,
                    "device": "cpu",
                    "dtype": "float32",
                    "width": 4,
                    "height": 4,
                },
                "defense": {"num_inference_steps": 2},
            }
        )
    )
    result = CliRunner().invoke(
        app,
        [
            "--model",
            "diffusers",
            "--model-config",
            str(config),
            "--attack",
            "ring_a_bell",
            "--defense",
            "trasce",
            "--prompt",
            "a dog sitting in a park",
            "--target",
            "dog",
            "--seed",
            "42",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert pipe.encodings == ["", "adversarial tokens", "dog"]
    assert len(judge_calls) == 1
    row = json.loads((tmp_path / "out/results.jsonl").read_text(encoding="utf-8"))
    assert row["target"] == "dog"
    assert row["defense_bypassed"] is True
    assert row["success"] is target_visible
    assert row["success_rule"] == "llm_target_presence"
    detail = json.loads((tmp_path / "out/details.jsonl").read_text(encoding="utf-8"))
    assert detail["metadata"]["image_defense"]["metadata"]["trasce_applied"]
