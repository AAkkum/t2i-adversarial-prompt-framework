from pathlib import Path

from t2i_framework.attacks.identity import IdentityAttack
from t2i_framework.core.types import PromptCase
from t2i_framework.defenses.none import NoneDefense
from t2i_framework.evaluation.runner import ExperimentRunner
from t2i_framework.models.mock_model import MockImageModel


def test_runner_records_prompt_case_metadata(tmp_path: Path) -> None:
    runner = ExperimentRunner(
        model=MockImageModel(),
        attack=IdentityAttack(),
        defense=NoneDefense(),
        output_dir=tmp_path,
    )

    results = runner.run(
        prompts=[
            PromptCase(
                prompt="a blue rabbit mascot",
                target_concept="blue rabbit mascot",
                case_id="case_001",
                category="synthetic",
                metadata={"notes": "debug row"},
            )
        ],
        seed=42,
        config={},
    )

    assert results[0].metadata["prompt_case"] == {
        "id": "case_001",
        "category": "synthetic",
        "notes": "debug row",
    }
    assert results[0].case_id == "case_001"
    assert results[0].category == "synthetic"
