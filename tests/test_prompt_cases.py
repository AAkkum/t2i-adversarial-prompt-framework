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


def test_read_prompt_file_supports_json_batches(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        """
        {
          "prompts": [
            {
              "id": "case_001",
              "prompt": "a blue rabbit mascot",
              "target": "blue rabbit mascot",
              "category": "synthetic"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    prompt_case = read_prompt_file(path)[0]

    assert prompt_case.case_id == "case_001"
    assert prompt_case.prompt == "a blue rabbit mascot"
    assert prompt_case.target_concept == "blue rabbit mascot"
    assert prompt_case.category == "synthetic"


def test_read_prompt_file_supports_jsonl_batches(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '{"prompt": "a red cube robot", "target_concept": "red cube robot"}\n'
        '"a green owl emblem"\n',
        encoding="utf-8",
    )

    prompt_cases = read_prompt_file(path)

    assert prompt_cases[0].target_concept == "red cube robot"
    assert prompt_cases[1].prompt == "a green owl emblem"
