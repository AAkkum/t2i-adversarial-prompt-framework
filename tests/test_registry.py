from t2i_framework.core.registry import available_components


def test_known_components_exist() -> None:
    components = available_components()
    assert {"mock", "diffusers"} <= set(components["models"])
    assert set(components["attacks"]) == {"daca", "groot", "identity", "pgj", "ring_a_bell"}
    assert set(components["defenses"]) == {
        "none",
        "latent_guard_lite",
        "safree",
        "trasce",
    }
    assert "clip_similarity" not in components["defenses"]
    assert "image_clip_filter" not in components["defenses"]
    assert "char_perturb" not in components["attacks"]
    assert "normalize_keywords" not in components["defenses"]
    assert "composite" not in components["defenses"]
