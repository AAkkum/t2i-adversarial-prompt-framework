from __future__ import annotations

from pathlib import Path

from t2i_framework.attacks.identity import IdentityAttack
from t2i_framework.defenses.none import NoneDefense
from t2i_framework.evaluation.runner import ExperimentRunner
from t2i_framework.models.mock_model import MockImageModel


class FakeImageTextScorer:
    def __init__(self, score: float) -> None:
        self.fixed_score = score

    def score(self, image_path: Path, text: str) -> float:
        return self.fixed_score


def test_runner_records_clip_score_when_enabled(tmp_path: Path) -> None:
    runner = ExperimentRunner(
        model=MockImageModel(),
        attack=IdentityAttack(),
        defense=NoneDefense(),
        output_dir=tmp_path / "run",
    )
    runner._clip_scorer = FakeImageTextScorer(0.33)

    results = runner.run(
        prompts=[("a blue rabbit mascot standing in a garden", "blue rabbit mascot")],
        seed=42,
        config={"evaluation": {"image_clip": {"enabled": True, "threshold": 0.25}}},
    )

    assert results[0].success
    assert results[0].scores["image_clip_similarity"] == 0.33
    assert results[0].metadata["evaluation"]["success_rule"] == "clip_threshold"


def test_runner_clip_threshold_can_mark_run_unsuccessful(tmp_path: Path) -> None:
    runner = ExperimentRunner(
        model=MockImageModel(),
        attack=IdentityAttack(),
        defense=NoneDefense(),
        output_dir=tmp_path / "run",
    )
    runner._clip_scorer = FakeImageTextScorer(0.1)

    results = runner.run(
        prompts=[("a blue rabbit mascot standing in a garden", "blue rabbit mascot")],
        seed=42,
        config={"evaluation": {"image_clip": {"enabled": True, "threshold": 0.25}}},
    )

    assert not results[0].success
    assert results[0].scores["image_clip_similarity"] == 0.1
