from scripts.run_latent_guard_all_4gpu import (
    ATTACKS,
    LATENT_GUARD_CONFIG,
    MODEL_PRESETS,
    build_latent_guard_cases,
)
from scripts.run_sdxl_safety_15_4gpu import _ends_attack_block


def test_full_latent_guard_matrix_covers_every_attack_and_model() -> None:
    cases = build_latent_guard_cases(list(MODEL_PRESETS))

    assert tuple(MODEL_PRESETS) == ("sd14", "sd35_medium")
    assert len(cases) == 10
    assert len({case.name for case in cases}) == 10
    assert all(case.model_label != "mock" for case in cases)

    expected = []
    for attack, max_candidates in ATTACKS:
        for model_name, model_config in MODEL_PRESETS.items():
            expected.append(
                (
                    model_name,
                    model_config,
                    attack,
                    "latent_guard_lite",
                    max_candidates,
                    LATENT_GUARD_CONFIG,
                )
            )

    assert [
        (
            case.model_label,
            case.model_config,
            case.attack,
            case.defense,
            case.max_candidates,
            case.defense_config,
        )
        for case in cases
    ] == expected


def test_single_model_subset_keeps_every_attack() -> None:
    cases = build_latent_guard_cases(["sd35_medium"])

    assert len(cases) == 5
    assert [case.attack for case in cases] == [attack for attack, _ in ATTACKS]
    assert all(case.model_label == "sd35_medium" for case in cases)
    assert all(case.defense == "latent_guard_lite" for case in cases)

    daca_indexes = [index + 1 for index, case in enumerate(cases) if case.attack == "daca"]
    assert _ends_attack_block(daca_indexes[-1], "daca", cases)


def test_unknown_or_duplicate_model_is_rejected() -> None:
    for model_names in (["unknown"], ["sd14", "sd14"]):
        try:
            build_latent_guard_cases(model_names)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid model selection: {model_names}")
