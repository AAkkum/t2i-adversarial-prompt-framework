from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from t2i_framework.attacks.base import Attack
from t2i_framework.attacks.daca import DACAAttack
from t2i_framework.attacks.groot import GrootAttack
from t2i_framework.attacks.identity import IdentityAttack
from t2i_framework.attacks.pgj import PGJAttack
from t2i_framework.attacks.search_attack import SearchAttack
from t2i_framework.attacks.textfooler_style import TextFoolerStyleAttack
from t2i_framework.defenses.base import Defense
from t2i_framework.defenses.character_filter import CharacterFilterDefense
from t2i_framework.defenses.latent_guard_lite import LatentGuardLiteDefense
from t2i_framework.defenses.none import NoneDefense
from t2i_framework.models.base import ImageModel
from t2i_framework.models.diffusers_model import DiffusersImageModel
from t2i_framework.models.mock_model import MockImageModel

T = TypeVar("T")

MODEL_REGISTRY: dict[str, Callable[..., ImageModel]] = {
    "mock": MockImageModel,
    "diffusers": DiffusersImageModel,
}

ATTACK_REGISTRY: dict[str, Callable[[], Attack]] = {
    "daca": DACAAttack,
    "identity": IdentityAttack,
    "textfooler_style": TextFoolerStyleAttack,
    "groot": GrootAttack,
    "search_attack": SearchAttack,
    "pgj": PGJAttack,
}

DEFENSE_REGISTRY: dict[str, Callable[[], Defense]] = {
    "none": NoneDefense,
    "character_filter": CharacterFilterDefense,
    "latent_guard_lite": LatentGuardLiteDefense,
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
