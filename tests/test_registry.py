from t2i_framework.core.registry import available_components


def test_known_components_exist() -> None:
    components = available_components()
    assert {"mock", "diffusers"} <= set(components["models"])
    assert {"identity", "char_perturb", "textfooler_style", "groot_lite", "search_attack"} <= set(
        components["attacks"]
    )
    assert {"none", "normalize_keywords", "embedding_filter", "image_clip_filter", "composite"} <= set(
        components["defenses"]
    )
