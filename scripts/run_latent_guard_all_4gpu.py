from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_sdxl_safety_15_4gpu import (  # noqa: E402
    REPO_ROOT,
    MatrixCase,
    main as run_matrix,
)


LATENT_GUARD_CONFIG = "configs/defenses/latent_guard_safety_nonsexual.yaml"
LATENT_GUARD_WEIGHTS = REPO_ROOT / "data/latent_guard/model_parameters.pth"

MODEL_PRESETS = {
    "sd14": "configs/models/sd14.yaml",
    "sd35_medium": "configs/models/sd35_medium.yaml",
}

ATTACKS = (
    ("identity", 1),
    ("pgj", 1),
    ("daca", 10),
    ("groot", 3),
    ("ring_a_bell", 1),
)


def build_latent_guard_cases(model_names: list[str]) -> tuple[MatrixCase, ...]:
    unknown = [name for name in model_names if name not in MODEL_PRESETS]
    if unknown:
        choices = ", ".join(MODEL_PRESETS)
        raise ValueError(
            f"Unknown model preset(s): {', '.join(unknown)}. Choose from: {choices}."
        )
    if len(set(model_names)) != len(model_names):
        raise ValueError("Each model preset may be selected only once.")

    cases: list[MatrixCase] = []
    for attack, max_candidates in ATTACKS:
        for model_name in model_names:
            model_config = MODEL_PRESETS[model_name]
            cases.append(
                MatrixCase(
                    number=f"{len(cases) + 1:02d}",
                    attack=attack,
                    defense="latent_guard_lite",
                    max_candidates=max_candidates,
                    defense_config=LATENT_GUARD_CONFIG,
                    model_config=model_config,
                    model_label=model_name,
                )
            )
    return tuple(cases)


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Comma-separated image-model presets to evaluate, or 'all'. "
            f"Available: {', '.join(MODEL_PRESETS)}."
        ),
    )


def _build_selected_cases(args: argparse.Namespace) -> tuple[MatrixCase, ...]:
    if not LATENT_GUARD_WEIGHTS.is_file():
        raise SystemExit(
            "Latent Guard checkpoint not found: "
            f"{LATENT_GUARD_WEIGHTS}. Upload model_parameters.pth before running."
        )

    raw_models = str(args.models).strip()
    model_names = (
        list(MODEL_PRESETS)
        if raw_models.casefold() == "all"
        else [item.strip() for item in raw_models.split(",") if item.strip()]
    )
    if not model_names:
        raise SystemExit("Select at least one model with --models.")
    try:
        return build_latent_guard_cases(model_names)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def main() -> None:
    run_matrix(
        cases=(),
        output_label="latent_guard",
        description=(
            "Evaluate Latent Guard against the paper attack set with SD 1.4 "
            "and SD 3.5 Medium, sharded across multiple GPUs."
        ),
        configure_parser=_configure_parser,
        case_builder=_build_selected_cases,
    )


if __name__ == "__main__":
    main()
