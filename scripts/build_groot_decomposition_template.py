from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Groot-lite decomposition template from a prompt CSV."
    )
    parser.add_argument("prompt_file", type=Path, help="CSV with prompt and target_concept columns.")
    parser.add_argument(
        "--existing",
        type=Path,
        default=Path("data/groot_decompositions.yaml"),
        help="Existing decomposition YAML used to mark known concepts.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output YAML template to write.",
    )
    args = parser.parse_args()

    targets = _read_targets(args.prompt_file)
    existing = _read_existing(args.existing)
    output = {
        "concepts": {
            target: {
                "status": "existing" if target.lower() in existing else "needs_decomposition",
                "decompositions": list(existing.get(target.lower(), []))
                or [
                    "TODO: add safe visual decomposition",
                    "TODO: add alternate safe visual decomposition",
                ],
            }
            for target in targets
        }
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(output, sort_keys=True), encoding="utf-8")


def _read_targets(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "target_concept" not in (reader.fieldnames or []):
            raise ValueError("Prompt CSV must contain a 'target_concept' column.")
        targets = {
            row["target_concept"].strip()
            for row in reader
            if row.get("target_concept") and row["target_concept"].strip()
        }
    return sorted(targets, key=str.lower)


def _read_existing(path: Path) -> dict[str, tuple[str, ...]]:
    if not path.exists():
        return {}
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    concepts = data.get("concepts", {})
    return {
        str(concept).strip().lower(): tuple(
            str(item).strip()
            for item in config.get("decompositions", [])
            if str(item).strip()
        )
        for concept, config in concepts.items()
        if isinstance(config, dict)
    }


if __name__ == "__main__":
    main()
