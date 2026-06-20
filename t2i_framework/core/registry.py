from __future__ import annotations

from typing import Any, Callable, TypeVar

from t2i_framework.attacks.base import Attack
from t2i_framework.attacks.char_perturb import CharPerturbAttack
from t2i_framework.attacks.groot_lite import GrootLiteAttack
from t2i_framework.attacks.identity import IdentityAttack
from t2i_framework.attacks.search_attack import SearchAttack
from t2i_framework.attacks.textfooler_style import TextFoolerStyleAttack
from t2i_framework.defenses.base import Defense
from t2i_framework.defenses.composite import CompositeDefense
from t2i_framework.defenses.embedding_filter import EmbeddingFilterDefense
from t2i_framework.defenses.image_clip_filter import ImageClipFilterDefense
from t2i_framework.defenses.none import NoneDefense
from t2i_framework.defenses.normalize_keywords import NormalizeKeywordsDefense
from t2i_framework.models.base import ImageModel
from t2i_framework.models.diffusers_model import DiffusersImageModel
from t2i_framework.models.mock_model import MockImageModel

T = TypeVar("T")

MODEL_REGISTRY: dict[str, Callable[..., ImageModel]] = {
    "mock": MockImageModel,
    "diffusers": DiffusersImageModel,
}

ATTACK_REGISTRY: dict[str, Callable[[], Attack]] = {
    "identity": IdentityAttack,
    "char_perturb": CharPerturbAttack,
    "textfooler_style": TextFoolerStyleAttack,
    "groot_lite": GrootLiteAttack,
    "search_attack": SearchAttack,
}

DEFENSE_REGISTRY: dict[str, Callable[[], Defense]] = {
    "none": NoneDefense,
    "normalize_keywords": NormalizeKeywordsDefense,
    "embedding_filter": EmbeddingFilterDefense,
    "image_clip_filter": ImageClipFilterDefense,
    "composite": CompositeDefense,
}


def available_components() -> dict[str, list[str]]:
    return {
        "models": sorted(MODEL_REGISTRY),
        "attacks": sorted(ATTACK_REGISTRY),
        "defenses": sorted(DEFENSE_REGISTRY),
    }


def _build(name: str, registry: dict[str, Callable[..., T]], kind: str, **kwargs: Any) -> T:
    try:
        return registry[name](**kwargs)
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown {kind} '{name}'. Available {kind}s: {available}") from exc


def build_model(name: str, **kwargs: Any) -> ImageModel:
    return _build(name, MODEL_REGISTRY, "model", **kwargs)


def build_attack(name: str) -> Attack:
    return _build(name, ATTACK_REGISTRY, "attack")


def build_defense(name: str) -> Defense:
    return _build(name, DEFENSE_REGISTRY, "defense")
