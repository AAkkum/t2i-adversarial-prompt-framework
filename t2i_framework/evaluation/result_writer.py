from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from t2i_framework.core.types import EvaluationResult


class ResultWriter:
    """Write compact results plus a separate full debugging trace."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "results.jsonl"
        self.details_path = self.output_dir / "details.jsonl"
        self.csv_path = self.output_dir / "results.csv"
        self._rows: list[dict[str, Any]] = []
        source_path = self.details_path if self.details_path.exists() else self.jsonl_path
        if source_path.exists():
            with source_path.open("r", encoding="utf-8") as handle:
                self._rows = [_json_safe(json.loads(line)) for line in handle if line.strip()]

    def append(self, result: EvaluationResult) -> None:
        row = _json_safe(asdict(result))
        self._rows.append(row)
        self._persist()

    def update(self, result: EvaluationResult) -> None:
        row = _json_safe(asdict(result))
        for index, previous in enumerate(self._rows):
            if previous["run_id"] == result.run_id:
                self._rows[index] = row
                self._persist()
                return
        raise KeyError(result.run_id)

    def _persist(self) -> None:
        summaries = [_summary_row(row) for row in self._rows]
        with _atomic_text(self.jsonl_path) as handle:
            for row in summaries:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        with _atomic_text(self.details_path) as handle:
            for row in self._rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self._write_csv(summaries)

    def _write_csv(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        base_fieldnames = list(rows[0].keys())
        score_fieldnames = sorted(
            {
                f"score_{score_name}"
                for row in rows
                for score_name in (row.get("scores") or {})
            }
        )
        fieldnames = [*base_fieldnames, *score_fieldnames]
        with _atomic_text(self.csv_path) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                csv_row = _csv_row(row)
                for score_name, score_value in (row.get("scores") or {}).items():
                    csv_row[f"score_{score_name}"] = score_value
                writer.writerow(csv_row)


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    evaluation = metadata.get("evaluation") or {}
    attack = metadata.get("attack_candidate") or {}
    scores = {
        key: value
        for key, value in (row.get("scores") or {}).items()
        if key not in {"candidate_score", "filter_pass_score", "text_similarity"}
    }
    status = "EVALUATION_ERROR" if metadata.get("evaluation_error") else metadata.get("status")
    return {
        "run_id": row.get("run_id"),
        "case_id": row.get("case_id"),
        "category": row.get("category"),
        "model": row.get("model_name"),
        "attack": row.get("attack_name"),
        "defense": row.get("defense_name"),
        "original_prompt": row.get("original_prompt"),
        "attacked_prompt": row.get("attacked_prompt"),
        "target": row.get("target_concept"),
        "seed": row.get("seed"),
        "candidate": _one_based_number(metadata.get("candidate_index")),
        "strategy": attack.get("strategy"),
        "prompt_blocked": row.get("prompt_blocked"),
        "image_blocked": row.get("image_blocked"),
        "defense_bypassed": metadata.get("defense_bypassed", False),
        "image": row.get("generated_image_path"),
        "success": row.get("success"),
        "success_rule": evaluation.get("success_rule", "pending"),
        "evaluation": evaluation.get("llm_judge", {"status": "pending"}),
        "scores": scores,
        "queries": row.get("query_count"),
        "runtime_seconds": round(float(row.get("runtime_seconds") or 0.0), 3),
        "status": status,
        "error": metadata.get("error") or metadata.get("evaluation_error"),
        "selected_best": bool(metadata.get("selected_best", False)),
        "final_image": metadata.get("final_image_path"),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _one_based_number(value: Any) -> Any:
    return value + 1 if isinstance(value, int) and not isinstance(value, bool) else value


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value, ensure_ascii=False, allow_nan=False)
        if isinstance(value, (dict, list))
        else value
        for key, value in row.items()
    }


@contextmanager
def _atomic_text(path: Path):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
