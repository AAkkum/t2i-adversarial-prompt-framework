from t2i_framework.models.diffusers_model import DiffusersImageModel


def test_diffusers_unload_releases_pipeline() -> None:
    model = DiffusersImageModel(unload_after_generation=True, log_unload=False)
    model._pipeline = object()  # noqa: SLF001

    model.unload()

    assert model._pipeline is None  # noqa: SLF001


def test_diffusers_keeps_pipeline_by_default() -> None:
    model = DiffusersImageModel()

    assert model.unload_after_generation is False
