from t2i_framework.core.registry import available_components


def test_known_components_exist() -> None:
    components = available_components()
    assert {"mock", "diffusers"} <= set(components["models"])
    assert {"daca", "identity", "textfooler_style", "groot", "search_attack", "pgj"} <= set(
        components["attacks"]
    )
    assert set(components["defenses"]) == {
        "none",
        "character_filter",
        "latent_guard_lite",
        "trasce",
    }
    assert "ring_a_bell" in components["attacks"]
    assert "clip_similarity" not in components["defenses"]
    assert "image_clip_filter" not in components["defenses"]
    assert "char_perturb" not in components["attacks"]
    assert "normalize_keywords" not in components["defenses"]
    assert "composite" not in components["defenses"]
