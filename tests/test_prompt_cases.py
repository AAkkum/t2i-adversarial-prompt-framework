from pathlib import Path

from t2i_framework.evaluation.prompt_cases import read_prompt_file


def test_read_prompt_file_supports_id_category_and_extra_metadata(tmp_path: Path) -> None:
    path = tmp_path / "prompts.csv"
    path.write_text(
        "\n".join(
            [
                "id,prompt,target_concept,category,notes",
                "case_001,a blue rabbit mascot,blue rabbit mascot,synthetic,debug row",
            ]
        ),
        encoding="utf-8",
    )

    prompt_case = read_prompt_file(path)[0]

    assert prompt_case.case_id == "case_001"
    assert prompt_case.prompt == "a blue rabbit mascot"
    assert prompt_case.target_concept == "blue rabbit mascot"
    assert prompt_case.category == "synthetic"
    assert prompt_case.metadata == {"notes": "debug row"}
