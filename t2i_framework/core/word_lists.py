from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_ATTACK_TERMS_PATH = Path("data/attack_terms.yaml")


def load_attack_terms(path: Path | None = None) -> dict[str, Any]:
    terms_path = path or DEFAULT_ATTACK_TERMS_PATH
    if not terms_path.exists():
        return {}
    with terms_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_term_set(section: str, path: Path | None = None) -> set[str]:
    values = load_attack_terms(path).get(section, [])
    if not isinstance(values, list):
        return set()
    return {str(value).lower() for value in values}


def load_replacement_map(section: str, path: Path | None = None) -> dict[str, str]:
    values = load_attack_terms(path).get(section, {})
    if not isinstance(values, dict):
        return {}
    return {str(key).lower(): str(value) for key, value in values.items()}
