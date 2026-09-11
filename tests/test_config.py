from pathlib import Path

from t2i_framework.core.config import default_component_config_path, merge_configs


def test_merge_configs_deep_merges_later_values() -> None:
    merged = merge_configs(
        {
            "model": {"name": "mock", "width": 512},
            "attack": {"candidate_count": 5},
        },
        {
            "model": {"width": 768},
            "defense": {"threshold": 0.7},
        },
    )
    assert merged == {
        "model": {"name": "mock", "width": 768},
        "attack": {"candidate_count": 5},
        "defense": {"threshold": 0.7},
    }


def test_default_component_config_path_finds_matching_yaml() -> None:
    path = default_component_config_path("attacks", "textfooler_style", Path("configs"))
    assert path == Path("configs") / "attacks" / "textfooler_style.yaml"
