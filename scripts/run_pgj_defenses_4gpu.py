from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_sdxl_safety_15_4gpu import MatrixCase, main as run_matrix


def build_cases() -> tuple[MatrixCase, ...]:
    return (
        MatrixCase(
            "01",
            "pgj",
            "none",
            1,
            model_config="configs/models/sdxl.yaml",
            model_label="sdxl",
            share_candidate_cache=False,
        ),
        MatrixCase(
            "02",
            "pgj",
            "safree",
            1,
            model_config="configs/models/sdxl.yaml",
            model_label="sdxl",
            share_candidate_cache=False,
        ),
        MatrixCase(
            "03",
            "pgj",
            "none",
            1,
            model_config="configs/models/sd14.yaml",
            model_label="sd14",
            share_candidate_cache=False,
        ),
        MatrixCase(
            "04",
            "pgj",
            "trasce",
            1,
            model_config="configs/models/sd14.yaml",
            model_label="sd14",
            share_candidate_cache=False,
        ),
    )


def main() -> None:
    run_matrix(
        cases=build_cases(),
        output_label="pgj_defenses",
        description=(
            "Evaluate PGJ against SAFREE and TraSCE with model-matched no-defense "
            "baselines and fresh candidates for every case. Latent Guard and "
            "CharacterFilter are excluded."
        ),
    )


if __name__ == "__main__":
    main()
