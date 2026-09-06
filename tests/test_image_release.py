"""Storage and export regressions; no learned models are loaded."""

import csv
import json
from pathlib import Path

import pytest

from t2i_framework.attacks.base import Attack
from t2i_framework.core.types import AttackCandidate, DefenseDecision
from t2i_framework.defenses.base import Defense
from t2i_framework.defenses.character_filter import CharacterFilterDefense
from t2i_framework.defenses.semantic_concepts import SemanticMatch
from t2i_framework.evaluation.runner import ExperimentRunner
from t2i_framework.models.mock_model import MockImageModel


class Candidates(Attack):
    name = "search_attack"

    def generate(self, prompt, target_concept=None, context=None):
        return [AttackCandidate(f"neutral scene {index}") for index in range(5)]


class InspectDefense(Defense):
    name = "inspect"

    def __init__(self, output, failure=None):
        self.output = output
        self.failure = failure
        self.paths = []

    def check_image(self, image_path, target_concept=None, context=None):
        assert image_path.is_file()
        assert not image_path.is_relative_to(self.output)
        index = context["candidate_index"]
        assert not (self.output / context["output_filename"]).exists()
        rows = read_rows(self.output)
        assert len(rows) == index + 1
        assert rows[-1]["metadata"]["status"] == "STARTED"
        self.paths.append(image_path)
        if index == 4 and self.failure:
            raise RuntimeError("injected image failure")
        return DefenseDecision(allowed=index != 1, reason="test decision")


def read_rows(output):
    def reject_constant(value):
        raise AssertionError(f"Non-standard JSON: {value}")

    return [
        json.loads(line, parse_constant=reject_constant)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_quarantine_progress_errors_and_standard_exports(tmp_path):
    output = tmp_path / "results"
    defense = InspectDefense(output, failure=True)
    results = ExperimentRunner(
        MockImageModel(), Candidates(), defense, output, max_candidates=5
    ).run([("neutral scene", "neutral scene")], 42)
    assert all(not path.exists() for path in defense.paths)
    assert not list((tmp_path / ".image_quarantine").iterdir())
    assert results[0].success
    assert results[1].image_blocked
    assert results[1].generated_image_path is None
    assert results[4].metadata["status"] == "ERROR"
    assert results[4].metadata["error_stage"] == "image_defense"
    assert results[4].generated_image_path is None
    rows = read_rows(output)
    assert len(rows) == 5
    assert rows[1]["scores"]["candidate_score"] is None
    assert rows[4]["scores"]["candidate_score"] is None
    assert results[1].scores["candidate_score"] == float("-inf")
    winner = next(result for result in results if result.metadata.get("selected_best"))
    assert winner.success
    assert Path(winner.metadata["final_image_path"]).read_bytes() == Path(
        winner.generated_image_path
    ).read_bytes()
    assert sum(bool(row["metadata"].get("selected_best")) for row in rows) == 1
    with (output / "results.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert json.loads(csv_rows[4]["scores"])["candidate_score"] is None


@pytest.mark.parametrize("stage", ["blip", "minilm_image"])
def test_real_defense_interface_fails_closed(tmp_path, stage):
    class Captioner:
        def caption(self, path):
            assert path.is_file()
            if stage == "blip":
                raise RuntimeError("BLIP test failure")
            return "caption: neutral scene"

    class Matcher:
        def match(self, text, concepts):
            if text.startswith("caption:"):
                raise RuntimeError("MiniLM test failure")
            return SemanticMatch("concept", "neutral", 0.1)

    output = tmp_path / "results"
    defense = CharacterFilterDefense(captioner=Captioner(), semantic_matcher=Matcher())
    results = ExperimentRunner(
        MockImageModel(), Candidates(), defense, output
    ).run([("neutral scene", None)], 42)
    assert results[0].metadata["error_stage"] == stage
    assert results[0].metadata["status"] == "ERROR"
    assert not results[0].success
    assert results[0].generated_image_path is None
    assert not list(output.glob("*.png"))
    assert not list((tmp_path / ".image_quarantine").iterdir())
    assert read_rows(output)[0]["metadata"]["status"] == "ERROR"


def test_generation_failure_preserves_started_candidate(tmp_path):
    class BrokenModel(MockImageModel):
        def generate(self, prompt, output_dir, seed, context=None):
            super().generate(prompt, output_dir, seed, context)
            raise RuntimeError("generation failed after writing")

    output = tmp_path / "results"
    results = ExperimentRunner(BrokenModel(), Candidates(), InspectDefense(output), output).run(
        [("neutral", None)], 42
    )
    assert results[0].metadata["error_stage"] == "generation"
    assert read_rows(output)[0]["metadata"]["status"] == "ERROR"
    assert not list(output.glob("*.png"))
    assert not list((tmp_path / ".image_quarantine").iterdir())


def test_release_failure_keeps_existing_image_and_logs_error(tmp_path, monkeypatch):
    import t2i_framework.evaluation.runner as runner_module

    output = tmp_path / "results"
    output.mkdir()
    existing = output / "previous_allowed.png"
    existing.write_bytes(b"existing allowed image")

    def fail_link(*args):
        raise OSError("release denied")

    monkeypatch.setattr(runner_module.os, "link", fail_link)
    result = ExperimentRunner(
        MockImageModel(), Candidates(), InspectDefense(output), output
    ).run([("neutral", None)], 42)[0]
    assert result.metadata["error_stage"] == "image_release"
    assert result.generated_image_path is None
    assert not result.success
    assert existing.read_bytes() == b"existing allowed image"
    assert list(output.glob("*.png")) == [existing]
    assert not list((tmp_path / ".image_quarantine").iterdir())


def test_export_sanitizes_all_nonfinite_values_without_mutating_scores(tmp_path):
    from t2i_framework.evaluation.result_writer import ResultWriter

    output = tmp_path / "results"
    result = ExperimentRunner(
        MockImageModel(), Candidates(), InspectDefense(output), output
    ).run([("neutral", None)], 42)[0]
    result.scores.update(nan=float("nan"), positive=float("inf"), negative=float("-inf"))
    writer = ResultWriter(output)
    writer.update(result)
    row = read_rows(output)[0]
    assert all(row["scores"][key] is None for key in ("nan", "positive", "negative"))
    assert result.scores["positive"] == float("inf")
    with (output / "results.csv").open(encoding="utf-8", newline="") as handle:
        csv_row = next(csv.DictReader(handle))
    assert json.loads(csv_row["scores"]) == row["scores"]
