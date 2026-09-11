from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from t2i_framework.attacks.search_attack import (
    SearchAttack,
    _build_variant_pool,
    _normalized_candidate,
)
from t2i_framework.defenses.character_filter import (
    SEMANTIC_IMAGE_THRESHOLD,
    SEMANTIC_PROMPT_THRESHOLD,
    CharacterFilterDefense,
    load_blocked_terms,
    load_protected_concepts,
)
from t2i_framework.defenses.semantic_concepts import MiniLMConceptMatcher, SemanticMatch
from t2i_framework.search_support import DATA_DIR, semantic_base

CASES = json.loads((DATA_DIR / "test_cases.json").read_text(encoding="utf-8"))


class FakeMatcher:
    def __init__(self, prompt_score: float = 0.1, image_score: float | None = None) -> None:
        self.prompt_score = prompt_score
        self.image_score = prompt_score if image_score is None else image_score
        self.calls: list[str] = []

    def match(self, text: str, concepts: dict[str, str]) -> SemanticMatch:
        self.calls.append(text)
        term = next(iter(concepts))
        score = self.image_score if text.startswith("caption:") else self.prompt_score
        return SemanticMatch(term, concepts[term], score)


class FakeCaptioner:
    def __init__(self, caption: str = "caption: neutral generated scene") -> None:
        self.value = caption
        self.calls: list[Path] = []

    def caption(self, image_path: Path) -> str:
        self.calls.append(image_path)
        return self.value


class SequencedImageMatcher(FakeMatcher):
    def __init__(self, image_scores: list[float]) -> None:
        super().__init__(prompt_score=0.1)
        self.image_scores = iter(image_scores)

    def match(self, text: str, concepts: dict[str, str]) -> SemanticMatch:
        self.calls.append(text)
        term = next(iter(concepts))
        score = next(self.image_scores) if text.startswith("caption:") else self.prompt_score
        return SemanticMatch(term, concepts[term], score)


class RaisingCaptioner(FakeCaptioner):
    def caption(self, image_path: Path) -> str:
        self.calls.append(image_path)
        raise RuntimeError("caption failed")


class CountingMockModel:
    name = "mock"

    def __init__(self) -> None:
        from t2i_framework.models.mock_model import MockImageModel

        self.delegate = MockImageModel()
        self.calls: list[str] = []

    def generate(self, prompt: str, output_dir: Path, seed: int, context=None):
        self.calls.append(prompt)
        return self.delegate.generate(prompt, output_dir, seed, context)


def make_defense(
    prompt_score: float = 0.1,
    image_score: float = 0.1,
    *,
    image: bool = True,
) -> CharacterFilterDefense:
    return CharacterFilterDefense(
        semantic_matcher=FakeMatcher(prompt_score, image_score),
        captioner=FakeCaptioner(),
        enable_image_semantic=image,
    )


def test_data_files_load_terms_and_concepts() -> None:
    terms = load_blocked_terms(DATA_DIR / "blocked_terms.txt")
    concepts = load_protected_concepts(DATA_DIR / "concept_targets.json")
    assert terms
    assert concepts
    for case in CASES[:-1]:
        assert case["blocked_term"] in terms
        assert concepts[case["blocked_term"]] == case["protected_concept"]


@pytest.mark.parametrize("case", CASES[:-1], ids=lambda case: case["id"])
def test_keyword_stage_blocks_original_and_reports_mapping(case) -> None:
    matcher = FakeMatcher(0.99)
    defense = CharacterFilterDefense(semantic_matcher=matcher, captioner=FakeCaptioner())
    for prompt in (case["prompt"], case["prompt"].upper()):
        decision = defense.check_prompt(prompt)
        assert not decision.allowed
        assert decision.metadata["matched_keyword_block"] == case["blocked_term"]
        assert decision.metadata["closest_protected_concept"] == case["protected_concept"]
        assert decision.metadata["blocked_by"] == "KEYWORD"
    assert matcher.calls == []  # Keyword blocking has precedence and skips MiniLM.


def test_control_passes_keyword_and_semantic_prompt_stages() -> None:
    decision = make_defense(prompt_score=0.20).check_prompt(CASES[-1]["prompt"])
    assert decision.allowed
    assert decision.metadata["keyword_result"] == "PASS"
    assert decision.metadata["semantic_prompt_result"] == "PASS"
    assert decision.metadata["blocked_by"] == "NONE"


def test_semantic_rephrasing_is_blocked_at_threshold() -> None:
    prompt = CASES[0]["protected_concept"] + " wearing blue overalls"
    decision = make_defense(prompt_score=SEMANTIC_PROMPT_THRESHOLD).check_prompt(prompt)
    assert not decision.allowed
    assert decision.metadata["direct_blocked_term"] is None
    assert decision.metadata["semantic_prompt_similarity"] == SEMANTIC_PROMPT_THRESHOLD
    assert decision.metadata["blocked_by"] == "MINILM_PROMPT"


def test_threshold_can_be_changed_without_changing_python_logic() -> None:
    defense = CharacterFilterDefense(
        semantic_prompt_threshold=0.75,
        semantic_matcher=FakeMatcher(0.70),
        captioner=FakeCaptioner(),
    )
    assert defense.check_prompt("a neutral reformulation").allowed
    with pytest.raises(ValueError, match="between 0 and 1"):
        CharacterFilterDefense(semantic_prompt_threshold=1.1)


def test_blip_caption_is_compared_and_can_block_image(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"mock")
    defense = make_defense(prompt_score=0.1, image_score=SEMANTIC_IMAGE_THRESHOLD)
    decision = defense.check_image(image, context={"candidate_index": 2})
    assert not decision.allowed
    assert decision.metadata["blip_caption"].startswith("caption:")
    assert decision.metadata["semantic_image_similarity"] == SEMANTIC_IMAGE_THRESHOLD
    assert decision.metadata["blocked_by"] == "BLIP_MINILM_IMAGE"


def test_disabled_stages_support_keyword_only_experiment(tmp_path: Path) -> None:
    defense = CharacterFilterDefense(
        enable_semantic_prompt=False,
        enable_image_semantic=False,
    )
    assert defense.check_prompt("a neutral scene").allowed
    assert defense.check_prompt("a neutral scene").metadata["semantic_prompt_result"] == "DISABLED"
    assert defense.check_image(tmp_path / "not-needed.png").allowed


def test_generic_blocklist_loading_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "terms.txt"
    path.write_text("\ufeff# comment\n\nTEST PHRASE\ntest phrase\n", encoding="utf-8")
    assert load_blocked_terms(path) == ("test phrase",)
    concepts = tmp_path / "concepts.json"
    concepts.write_text('{"TEST PHRASE": "described subject"}', encoding="utf-8")
    defense = CharacterFilterDefense(
        path,
        concepts,
        enable_semantic_prompt=False,
        enable_image_semantic=False,
    )
    assert not defense.check_prompt("test   phrase").allowed
    assert defense.check_prompt("test phrasebook").allowed
    path.write_text("", encoding="utf-8")
    assert CharacterFilterDefense(
        path, concepts, enable_semantic_prompt=False, enable_image_semantic=False
    ).check_prompt("test phrase").allowed


def test_missing_data_files_fail_visibly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CharacterFilterDefense(tmp_path / "missing.txt")
    with pytest.raises(FileNotFoundError):
        CharacterFilterDefense(concepts_path=tmp_path / "missing.json")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("count", [1, 5, 10, 20])
def test_candidate_counts_original_target_and_unique(case, count) -> None:
    original = "  " + case["prompt"] + "  "
    candidates = SearchAttack().generate(
        original, case["target"], {"seed": case["seed"], "max_candidates": count}
    )
    assert len(candidates) == count
    assert candidates[0].text == original
    normalized = [_normalized_candidate(item.text) for item in candidates]
    assert len(set(normalized)) == count
    keyword_only = CharacterFilterDefense(
        enable_semantic_prompt=False, enable_image_semantic=False
    )
    for item in candidates[1:]:
        assert case["target"] in item.text
        assert keyword_only.check_prompt(item.text).allowed
        assert len(item.text.split()) <= 80


@pytest.mark.parametrize("count", [0, -1, 21, True, 1.5])
def test_bad_candidate_count(count) -> None:
    with pytest.raises(ValueError, match="between 1 and 20"):
        SearchAttack().generate(CASES[0]["prompt"], context={"max_candidates": count})


def test_seed_and_pool_diversity() -> None:
    pools = [_build_variant_pool(CASES[1]["target"], "in a park", seed) for seed in (42, 7, 42)]
    assert pools[0] == pools[2]
    assert pools[0] != pools[1]
    for pool in pools:
        assert len(pool) == 400
        assert len({_normalized_candidate(text) for text, _ in pool}) == 400
        for _, categories in pool:
            assert 2 <= len(categories) <= 4
            assert len(set(categories)) == len(categories)
            assert not {"perspective", "camera_angle"} <= set(categories)
            assert not {"background", "depth_of_field"} <= set(categories)


def test_scene_and_data_mapping_preserve_original_action() -> None:
    mapping = load_protected_concepts(DATA_DIR / "concept_targets.json")
    assert "sitting beside a person in a park" in semantic_base(
        CASES[1]["prompt"], CASES[1]["target"], mapping
    )
    assert "parked beside a university building" in semantic_base(
        CASES[2]["prompt"], CASES[2]["target"], mapping
    )
    assert "working in a modern laboratory" in semantic_base(
        CASES[3]["prompt"], CASES[3]["target"], mapping
    )


def test_runner_skips_generation_for_prompt_blocked_candidates(tmp_path: Path) -> None:
    from t2i_framework.evaluation.runner import ExperimentRunner

    model = CountingMockModel()
    results = ExperimentRunner(
        model, SearchAttack(), make_defense(prompt_score=0.90), tmp_path, max_candidates=10
    ).run([(CASES[0]["prompt"], CASES[0]["target"])], seed=42)
    assert len(results) == 10
    assert model.calls == []
    assert all(result.prompt_blocked for result in results)
    assert not list(tmp_path.glob("*.png"))


def test_runner_uses_existing_image_hook_and_selects_allowed_winner(tmp_path: Path) -> None:
    from t2i_framework.evaluation.runner import ExperimentRunner

    model = CountingMockModel()
    captioner = FakeCaptioner()
    defense = CharacterFilterDefense(
        semantic_matcher=FakeMatcher(0.10, 0.10), captioner=captioner
    )
    results = ExperimentRunner(
        model, SearchAttack(), defense, tmp_path, max_candidates=10
    ).run([(CASES[-1]["prompt"], CASES[-1]["target"])], seed=42)
    assert len(model.calls) == 10
    assert len(captioner.calls) == 10
    assert all(result.success for result in results)
    winner = next(result for result in results if result.metadata.get("selected_best"))
    assert Path(winner.metadata["final_image_path"]).is_file()


def test_runner_image_block_prevents_winner(tmp_path: Path) -> None:
    from t2i_framework.evaluation.runner import ExperimentRunner

    model = CountingMockModel()
    results = ExperimentRunner(
        model,
        SearchAttack(),
        make_defense(prompt_score=0.10, image_score=0.90),
        tmp_path,
        max_candidates=2,
    ).run([(CASES[-1]["prompt"], CASES[-1]["target"])], seed=42)
    assert len(model.calls) == 2
    assert all(result.image_blocked for result in results)
    assert all(result.generated_image_path is None for result in results)
    assert all(result.metadata["blocked_by"] == "BLIP_MINILM_IMAGE" for result in results)
    assert all(result.metadata["image_disposition"] == "deleted_after_image_block" for result in results)
    assert all(result.metadata["image_defense"]["metadata"]["blip_caption"] for result in results)
    assert all(
        result.metadata["image_defense"]["metadata"]["semantic_image_similarity"] == 0.90
        for result in results
    )
    assert all(not Path(result.metadata["discarded_image_path"]).exists() for result in results)
    assert not any(result.metadata.get("selected_best") for result in results)
    assert not list(tmp_path.glob("*.png"))


def test_runner_never_selects_or_keeps_blocked_image(tmp_path: Path) -> None:
    from t2i_framework.evaluation.runner import ExperimentRunner

    model = CountingMockModel()
    defense = CharacterFilterDefense(
        semantic_matcher=SequencedImageMatcher([0.90, 0.10]),
        captioner=FakeCaptioner(),
    )
    results = ExperimentRunner(
        model, SearchAttack(), defense, tmp_path, max_candidates=2
    ).run([(CASES[-1]["prompt"], CASES[-1]["target"])], seed=42)

    blocked, allowed = results
    assert blocked.image_blocked
    assert blocked.generated_image_path is None
    assert not Path(blocked.metadata["discarded_image_path"]).exists()
    assert not blocked.metadata.get("selected_best")
    assert allowed.success
    assert Path(allowed.generated_image_path).is_file()
    assert allowed.metadata["selected_best"]
    assert Path(allowed.metadata["final_image_path"]).is_file()
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert rows[0]["generated_image_path"] is None
    assert rows[0]["metadata"]["image_defense"]["metadata"]["semantic_image_similarity"] == 0.90
    assert json.loads(csv_rows[0]["metadata"]) == rows[0]["metadata"]


def test_runner_removes_generated_image_when_image_defense_raises(tmp_path: Path) -> None:
    from t2i_framework.evaluation.runner import ExperimentRunner

    model = CountingMockModel()
    defense = CharacterFilterDefense(
        semantic_matcher=FakeMatcher(0.10, 0.10),
        captioner=RaisingCaptioner(),
    )
    runner = ExperimentRunner(model, SearchAttack(), defense, tmp_path, max_candidates=1)
    existing = tmp_path / "step_00_original_seed42.png"
    existing.write_bytes(b"existing allowed image")

    results = runner.run([(CASES[-1]["prompt"], CASES[-1]["target"])], seed=42)
    assert results[0].metadata["status"] == "ERROR"
    assert results[0].metadata["error"] == "caption failed"
    assert list(tmp_path.glob("*.png")) == [existing]
    assert existing.read_bytes() == b"existing allowed image"


@pytest.mark.parametrize("missing_path", [None, Path("missing-generated-image.png")])
def test_runner_rejects_missing_generation(tmp_path, missing_path):
    from t2i_framework.evaluation.runner import ExperimentRunner

    model = SimpleNamespace(
        name="test", generate=lambda *args, **kwargs: SimpleNamespace(image_path=missing_path)
    )
    results = ExperimentRunner(model, SearchAttack(), make_defense(), tmp_path).run(
        [(CASES[-1]["prompt"], CASES[-1]["target"])], seed=42
    )
    assert "no existing image" in results[0].metadata["error"]
    assert not list(tmp_path.glob("*.png"))


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_minilm_rejects_nonfinite_scores(monkeypatch, score):
    matcher = MiniLMConceptMatcher()
    monkeypatch.setattr(matcher, "_similarities", lambda *args: [score])
    with pytest.raises(RuntimeError, match="invalid similarity"):
        matcher.match("neutral scene", {"concept": "description"})


def test_terminal_output_explains_prompt_and_image_stages(tmp_path: Path, capsys) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"mock")
    defense = make_defense(prompt_score=0.75, image_score=0.80)
    defense.check_prompt("a natural semantic description", context={"candidate_index": 1})
    defense.check_image(image, context={"candidate_index": 1})
    output = capsys.readouterr().out
    for label in (
        "Direct blocked term:",
        "Closest protected concept:",
        "MiniLM prompt similarity:",
        "MiniLM threshold:",
        "BLIP Caption:",
        "MiniLM image similarity:",
        "Blocked By:",
        "Final Defense:",
    ):
        assert label in output


def test_minilm_masked_pooling_cosine_and_closest_concept() -> None:
    import torch

    matcher = MiniLMConceptMatcher()
    matcher._tokenizer = lambda *args, **kwargs: {
        "attention_mask": torch.tensor([[1, 0], [1, 0], [1, 0]])
    }
    matcher._model = lambda **kwargs: SimpleNamespace(last_hidden_state=torch.tensor([
        [[3.0, 0.0], [0.0, 99.0]],
        [[8.0, 0.0], [0.0, 99.0]],
        [[0.0, 4.0], [99.0, 0.0]],
    ]))
    match = matcher.match("reference", {"first": "matching", "second": "orthogonal"})
    assert match == SemanticMatch("first", "matching", 1.0)


def test_no_data_terms_hardcoded_and_no_clip_or_attack_scoring() -> None:
    sources = [
        DATA_DIR.parents[1] / "t2i_framework/defenses/character_filter.py",
        DATA_DIR.parents[1] / "t2i_framework/defenses/semantic_concepts.py",
        DATA_DIR.parents[1] / "t2i_framework/defenses/blip_caption.py",
    ]
    terms = load_blocked_terms(DATA_DIR / "blocked_terms.txt")
    for path in sources:
        source = path.read_text(encoding="utf-8")
        assert "clip" not in source.casefold().replace("blip", "")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                normalized = _normalized_candidate(node.value)
                assert all(term not in normalized.split() for term in terms)
    attack_source = (DATA_DIR.parents[1] / "t2i_framework/attacks/search_attack.py").read_text()
    assert "MiniLM" not in attack_source
    assert "BLIP" not in attack_source
    assert "final_score" not in attack_source
