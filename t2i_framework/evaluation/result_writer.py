from __future__ import annotations

import csv
import json
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
                self._rows = [json.loads(line) for line in handle if line.strip()]

    def append(self, result: EvaluationResult) -> None:
        row = asdict(result)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._rows.append(row)
        self._write_csv()

    def _write_csv(self) -> None:
        if not self._rows:
            return
        fieldnames = list(self._rows[0].keys())
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                        for key, value in row.items()
                    }
                )
