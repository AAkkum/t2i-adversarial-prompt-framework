from pathlib import Path

from t2i_framework.core.word_lists import load_replacement_map, load_term_set


def test_load_term_set_lowercases_list_values(tmp_path: Path) -> None:
    path = tmp_path / "terms.yaml"
    path.write_text("actions:\n  - Fighting\n  - standing\n", encoding="utf-8")

    assert load_term_set("actions", path) == {"fighting", "standing"}


def test_load_replacement_map_lowercases_keys(tmp_path: Path) -> None:
    path = tmp_path / "terms.yaml"
    path.write_text(
        "replacements:\n  Blue Rabbit Mascot: azure long-eared cartoon character\n",
        encoding="utf-8",
    )

    assert load_replacement_map("replacements", path) == {
        "blue rabbit mascot": "azure long-eared cartoon character"
    }
