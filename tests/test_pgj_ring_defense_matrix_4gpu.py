from scripts.run_pgj_ring_all_defenses_4gpu import ATTACKS, build_cases


def test_pgj_ring_matrix_covers_compatible_defenses_without_latent_guard() -> None:
    cases = build_cases()

    assert ATTACKS == ("pgj", "ring_a_bell")
    assert len(cases) == 10
    assert len({case.name for case in cases}) == 10

    expected_per_attack = [
        ("sdxl", "configs/models/sdxl.yaml", "none"),
        ("sdxl", "configs/models/sdxl.yaml", "character_filter"),
        ("sdxl", "configs/models/sdxl.yaml", "safree"),
        ("sd14", "configs/models/sd14.yaml", "none"),
        ("sd14", "configs/models/sd14.yaml", "trasce"),
    ]
    for attack_index, attack in enumerate(ATTACKS):
        block = cases[attack_index * 5 : (attack_index + 1) * 5]
        assert all(case.attack == attack for case in block)
        assert [
            (case.model_label, case.model_config, case.defense) for case in block
        ] == expected_per_attack
        assert all(case.max_candidates == 1 for case in block)

    assert all(case.defense != "latent_guard_lite" for case in cases)
