from __future__ import annotations

import csv
from pathlib import Path

from t2i_framework.core.types import PromptCase


RESERVED_COLUMNS = {"prompt", "target_concept", "target", "id", "case_id", "category"}


def read_prompt_file(path: Path) -> list[PromptCase]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "prompt" not in (reader.fieldnames or []):
            raise ValueError("Prompt CSV must contain a 'prompt' column.")
        return [_prompt_case_from_row(row) for row in reader]


def _prompt_case_from_row(row: dict[str, str]) -> PromptCase:
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
