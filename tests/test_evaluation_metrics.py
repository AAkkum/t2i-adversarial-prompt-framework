from pathlib import Path

from t2i_framework.attacks.identity import IdentityAttack
from t2i_framework.core.types import PromptCase
from t2i_framework.defenses.none import NoneDefense
from t2i_framework.evaluation.metrics import score_band
from t2i_framework.evaluation.prompt_similarity import PromptSimilarityEvaluator
from t2i_framework.evaluation.runner import ExperimentRunner
from t2i_framework.models.mock_model import MockImageModel


class FakePromptSimilarityScorer:
    def score(self, source: str, candidate: str) -> float:
        assert source == "a woman studying"
        assert candidate == "a person reading notes"
        return 0.82


def test_prompt_similarity_evaluator_records_prompt_prompt_similarity() -> None:
    evaluator = PromptSimilarityEvaluator(scorer=FakePromptSimilarityScorer())

    scores, metadata = evaluator.evaluate(
        "a woman studying",
        "a person reading notes",
        {
            "prompt_similarity": {
                "enabled": True,
                "bands": {
                    "strong": 0.8,
                    "moderate": 0.65,
                    "weak": 0.5,
                },
            }
        },
    )

    assert scores == {"prompt_prompt_similarity": 0.82}
    assert metadata["prompt_similarity"]["quality_band"] == "strong"


def test_prompt_similarity_evaluator_can_disable_prompt_prompt_similarity() -> None:
    evaluator = PromptSimilarityEvaluator()

    scores, metadata = evaluator.evaluate(
        "a woman studying",
        "a person reading notes",
        {"prompt_similarity": {"enabled": False}},
    )

    assert scores == {}
    assert metadata == {"prompt_similarity": {"enabled": False}}


def test_prompt_similarity_bands() -> None:
    bands = {"strong": 0.8, "moderate": 0.65, "weak": 0.5}

    assert score_band(0.82, bands) == "strong"
    assert score_band(0.667, bands) == "moderate"
    assert score_band(0.51, bands) == "weak"
    assert score_band(0.49, bands) == "drift"


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
        config={"evaluation": {"prompt_similarity": {"enabled": False}}},
    )

    assert results[0].metadata["prompt_case"] == {
        "id": "case_001",
        "category": "synthetic",
        "notes": "debug row",
    }
    assert results[0].case_id == "case_001"
    assert results[0].category == "synthetic"
