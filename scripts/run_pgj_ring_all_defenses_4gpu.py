from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_sdxl_safety_15_4gpu import MatrixCase, main as run_matrix


ATTACKS = ("pgj", "ring_a_bell")


def build_cases() -> tuple[MatrixCase, ...]:
    cases: list[MatrixCase] = []
    for attack in ATTACKS:
        for model_label, model_config, defense in (
            ("sdxl", "configs/models/sdxl.yaml", "none"),
            ("sdxl", "configs/models/sdxl.yaml", "character_filter"),
            ("sdxl", "configs/models/sdxl.yaml", "safree"),
            ("sd14", "configs/models/sd14.yaml", "none"),
            ("sd14", "configs/models/sd14.yaml", "trasce"),
        ):
            cases.append(
                MatrixCase(
                    number=f"{len(cases) + 1:02d}",
                    attack=attack,
                    defense=defense,
                    max_candidates=1,
                    model_config=model_config,
                    model_label=model_label,
                )
            )
    return tuple(cases)


def main() -> None:
    run_matrix(
        cases=build_cases(),
        output_label="pgj_ring_defenses",
        description=(
            "Evaluate PGJ and Ring-A-Bell against none, CharacterFilter, SAFREE, "
            "and TraSCE across their compatible models. Candidates are generated "
            "fresh for this matrix and reused across its defense cases."
        ),
    )


if __name__ == "__main__":
    main()
