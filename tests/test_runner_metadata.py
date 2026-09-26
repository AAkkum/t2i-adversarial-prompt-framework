import json
from pathlib import Path

from t2i_framework.attacks.base import Attack
from t2i_framework.attacks.identity import IdentityAttack
from t2i_framework.core.types import AttackCandidate, PromptCase
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
    assert results[0].run_id.startswith("run_0001_")
    assert Path(results[0].generated_image_path or "").name == "case_0001_seed42.png"


def test_batch_image_names_include_case_and_candidate_indexes(tmp_path: Path) -> None:
    class TwoCandidates(Attack):
        name = "two_candidates"

        def generate(self, prompt, target_concept=None, context=None):
            return [
                AttackCandidate(text=f"{prompt} version 0"),
                AttackCandidate(text=f"{prompt} version 1"),
            ]

    runner = ExperimentRunner(
        model=MockImageModel(),
        attack=TwoCandidates(),
        defense=NoneDefense(),
        output_dir=tmp_path,
        max_candidates=2,
    )

    results = runner.run(
        prompts=[("first prompt", "first"), ("second prompt", "second")],
        seed=7,
        config={},
    )

    assert [Path(result.generated_image_path or "").name for result in results] == [
        "case_0001_candidate_01_seed7.png",
        "case_0001_candidate_02_seed7.png",
        "case_0002_candidate_01_seed7.png",
        "case_0002_candidate_02_seed7.png",
    ]
    assert [result.run_id.split("_candidate_")[1] for result in results] == [
        "01",
        "02",
        "01",
        "02",
    ]
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["candidate"] for row in rows] == [1, 2, 1, 2]
