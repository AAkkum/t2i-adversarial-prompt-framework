from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for config in configs:
        merged = _deep_merge(merged, config)
    return merged


def load_merged_yaml_configs(*paths: Path | None) -> dict[str, Any]:
    return merge_configs(*(load_yaml_config(path) for path in paths))


def default_component_config_path(
    kind: str,
    name: str,
    config_dir: Path = Path("configs"),
) -> Path | None:
    path = config_dir / kind / f"{name}.yaml"
    return path if path.exists() else None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def save_yaml_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
