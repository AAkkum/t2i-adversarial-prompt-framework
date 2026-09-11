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
    """Append JSONL rows and maintain a CSV copy of experiment results."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "results.jsonl"
        self.csv_path = self.output_dir / "results.csv"
        self._rows: list[dict[str, Any]] = []
        if self.jsonl_path.exists():
            with self.jsonl_path.open("r", encoding="utf-8") as handle:
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
        with _atomic_text(self.jsonl_path) as handle:
            for row in self._rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self._write_csv()

    def _write_csv(self) -> None:
        if not self._rows:
            return
        fieldnames = list(self._rows[0].keys())
        with _atomic_text(self.csv_path) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                        if isinstance(value, (dict, list))
                        else value
                        for key, value in row.items()
                    }
                )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


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
