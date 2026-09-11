from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from t2i_framework.core.types import PromptCase


RESERVED_COLUMNS = {"prompt", "target_concept", "target", "id", "case_id", "category"}


def read_prompt_file(path: Path) -> list[PromptCase]:
    if path.suffix.lower() == ".json":
        return _read_json_prompt_file(path)
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl_prompt_file(path)
    return _read_csv_prompt_file(path)


def _read_csv_prompt_file(path: Path) -> list[PromptCase]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "prompt" not in (reader.fieldnames or []):
            raise ValueError("Prompt CSV must contain a 'prompt' column.")
        return [_prompt_case_from_row(row) for row in reader]


def _read_json_prompt_file(path: Path) -> list[PromptCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("prompts") or raw.get("cases")
    if not isinstance(raw, list):
        raise ValueError("Prompt JSON must contain a list, or an object with 'prompts'/'cases'.")
    return [_prompt_case_from_row(_json_row(item)) for item in raw]


def _read_jsonl_prompt_file(path: Path) -> list[PromptCase]:
    rows = [
        _json_row(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [_prompt_case_from_row(row) for row in rows]


def _json_row(item: Any) -> dict[str, str]:
    if isinstance(item, str):
        return {"prompt": item}
    if not isinstance(item, dict):
        raise ValueError("Each prompt JSON item must be an object or prompt string.")
    return {str(key): "" if value is None else str(value) for key, value in item.items()}


def _prompt_case_from_row(row: dict[str, str]) -> PromptCase:
    if not row.get("prompt"):
        raise ValueError("Each prompt case must contain a non-empty 'prompt' value.")
    metadata = {
        key: value
        for key, value in row.items()
        if key not in RESERVED_COLUMNS and value not in (None, "")
    }
    return PromptCase(
        prompt=row["prompt"],
        target_concept=row.get("target_concept") or row.get("target") or None,
        case_id=row.get("case_id") or row.get("id") or None,
        category=row.get("category") or None,
        metadata=metadata,
    )
