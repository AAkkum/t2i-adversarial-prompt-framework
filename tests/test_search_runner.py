from __future__ import annotations

from pathlib import Path

from t2i_framework.attacks.search_attack import SearchAttack
from t2i_framework.defenses.character_filter import CharacterFilterDefense
from t2i_framework.defenses.filter_placeholder import FilterPlaceholderDefense
from t2i_framework.evaluation.runner import ExperimentRunner, safe_prompt_slug
from t2i_framework.models.mock_model import MockImageModel


class CountingMockModel(MockImageModel):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str, output_dir: Path, seed: int, context=None):
        self.calls.append(prompt)
        return super().generate(prompt, output_dir, seed, context)


def test_search_runner_generates_allowed_candidates_and_selects_best(tmp_path: Path) -> None:
    model = CountingMockModel()
    runner = ExperimentRunner(
        model=model,
        attack=SearchAttack(),
        defense=CharacterFilterDefense(),
        output_dir=tmp_path,
        max_candidates=10,
    )
    results = runner.run(
        [
            (
                "Mario standing in a modern university laboratory",
                "red-capped mustached plumber character",
            )
        ],
        seed=42,
    )

    assert len(results) == 10
    assert len(model.calls) == 9
    assert sum(result.prompt_blocked for result in results) == 1
    assert results[0].prompt_blocked
    assert results[0].generated_image_path is None
    assert results[0].scores["candidate_score"] == float("-inf")
    assert any(result.success for result in results[1:])
    selected = [result for result in results if result.metadata.get("selected_best")]
    assert len(selected) == 1
    assert selected[0] is not results[0]
    assert Path(selected[0].metadata["final_image_path"]).exists()
    assert all(result.attack_name == "search_attack" for result in results)


def test_search_image_names_and_no_overwrite(tmp_path: Path) -> None:
    kwargs = {
        "model": MockImageModel(),
        "attack": SearchAttack(),
        "defense": FilterPlaceholderDefense(),
        "output_dir": tmp_path,
        "max_candidates": 2,
    }
    ExperimentRunner(**kwargs).run([("a blue robot", None)], seed=42)
    ExperimentRunner(**kwargs).run([("a blue robot", None)], seed=42)
    assert (tmp_path / "step_00_original_seed42.png").exists()
    assert (tmp_path / "step_00_original_seed42_01.png").exists()
    assert (tmp_path / "final_best_candidate_seed42.png").exists()
    assert (tmp_path / "final_best_candidate_seed42_01.png").exists()


def test_safe_prompt_slug_is_windows_safe_and_bounded() -> None:
    slug = safe_prompt_slug('Blue: rabbit? * in <lab> / "test"', max_length=24)
    assert slug == "blue-rabbit-in-lab-test"
    assert len(slug) <= 24
