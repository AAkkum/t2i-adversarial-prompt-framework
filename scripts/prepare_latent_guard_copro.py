from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from t2i_framework.evaluation.latent_guard_copro import concepts_for_split, load_copro


def write_concepts(path: Path, concepts: list[str], source: Path, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Generated from {source.as_posix()} ({split.upper()}_concepts).\n"
        "# Do not add aliases for the paper reproduction protocol.\n"
    )
    body = yaml.safe_dump(
        {"concepts": {concept: {} for concept in concepts}},
        sort_keys=False,
        allow_unicode=False,
    )
    path.write_text(header + body, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract official CoPro concept blacklists.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/latent_guard/CoPro_v1.0.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/latent_guard"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_copro(args.dataset)
    for split in ("id", "ood"):
        concepts = concepts_for_split(data, split)
        output = args.output_dir / f"restricted_concepts_copro_{split}.yaml"
        write_concepts(output, concepts, args.dataset, split)
        print(f"Wrote {len(concepts)} {split.upper()} concepts to {output}")


if __name__ == "__main__":
    main()
