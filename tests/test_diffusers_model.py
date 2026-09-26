from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from t2i_framework.models.diffusers_model import DiffusersImageModel


class _FakeImage:
    def save(self, path: Path) -> None:
        path.write_bytes(b"image")


class _FakePipeline:
    def __init__(self) -> None:
        self.to_calls: list[str] = []

    def to(self, device: str):
        self.to_calls.append(device)
        return self

    def __call__(self, **kwargs):
        return SimpleNamespace(images=[_FakeImage()])


def _install_fake_model_modules(monkeypatch):
    captured: dict[str, object] = {}
    pipeline = _FakePipeline()

    class _Generator:
        def __init__(self, device: str) -> None:
            captured["generator_device"] = device

        def manual_seed(self, seed: int):
            captured["seed"] = seed
            return self

    class _AutoPipeline:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            captured["model_id"] = model_id
            captured["load_kwargs"] = kwargs
            return pipeline

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    torch.float16 = object()
    torch.Generator = _Generator
    diffusers = ModuleType("diffusers")
    diffusers.AutoPipelineForText2Image = _AutoPipeline
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    monkeypatch.setitem(__import__("sys").modules, "diffusers", diffusers)
    return captured, pipeline, torch


def test_device_map_loads_pipeline_without_followup_move(tmp_path, monkeypatch) -> None:
    captured, pipeline, torch = _install_fake_model_modules(monkeypatch)
    model = DiffusersImageModel(
        model_id="example/sdxl",
        dtype="float16",
        device_map="cuda",
        low_cpu_mem_usage=True,
    )

    result = model.generate("test prompt", tmp_path, seed=17)

    assert captured["load_kwargs"] == {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": "cuda",
    }
    assert captured["generator_device"] == "cuda"
    assert pipeline.to_calls == []
    assert result.image_path.exists()
    assert result.metadata["device_map"] == "cuda"


def test_device_map_cannot_be_combined_with_cpu_offload() -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        DiffusersImageModel(device_map="cuda", enable_model_cpu_offload=True)


def test_dpm_solver_scheduler_override(tmp_path, monkeypatch) -> None:
    captured, pipeline, _torch = _install_fake_model_modules(monkeypatch)
    native_scheduler = SimpleNamespace(config={"name": "native"})
    pipeline.scheduler = native_scheduler
    diffusers = __import__("sys").modules["diffusers"]

    class _DPMSolver:
        @classmethod
        def from_config(cls, config):
            captured["scheduler_config"] = config
            return cls()

    diffusers.DPMSolverMultistepScheduler = _DPMSolver
    model = DiffusersImageModel(
        model_id="example/sdxl",
        dtype="float16",
        device_map="cuda",
        scheduler="dpm_solver",
    )
    model.generate("test prompt", tmp_path, seed=17)

    assert captured["scheduler_config"] == native_scheduler.config
    assert isinstance(pipeline.scheduler, _DPMSolver)
