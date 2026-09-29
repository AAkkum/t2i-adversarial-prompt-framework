from scripts.run_pgj_defenses_4gpu import build_cases


def test_pgj_matrix_covers_final_compatible_defenses() -> None:
    cases = build_cases()

    assert [
        (case.model_label, case.model_config, case.attack, case.defense)
        for case in cases
    ] == [
        ("sdxl", "configs/models/sdxl.yaml", "pgj", "none"),
        ("sdxl", "configs/models/sdxl.yaml", "pgj", "safree"),
        ("sdxl", "configs/models/sdxl.yaml", "pgj", "latent_guard_lite"),
        ("sd14", "configs/models/sd14.yaml", "pgj", "none"),
        ("sd14", "configs/models/sd14.yaml", "pgj", "trasce"),
    ]
    assert all(case.max_candidates == 1 for case in cases)
    assert all(case.share_candidate_cache is False for case in cases)
    assert cases[2].defense_config == (
        "configs/defenses/latent_guard_lite.yaml"
    )
