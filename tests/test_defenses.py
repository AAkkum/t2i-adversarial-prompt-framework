from pathlib import Path

from t2i_framework.defenses.clip_similarity import CLIPSimilarityDefense
from t2i_framework.defenses.image_clip_filter import ImageClipFilterDefense
from t2i_framework.defenses.none import NoneDefense
from t2i_framework.defenses.normalize_keywords import NormalizeKeywordsDefense


class FakeImageTextScorer:
    def __init__(self, score: float) -> None:
        self.fixed_score = score

    def score(self, image_path: Path, text: str) -> float:
        return self.fixed_score


def fake_clip_text_scorer(_prompt: str, terms: list[str]) -> list[float]:
    return [0.8 if term == "blue rabbit mascot" else 0.1 for term in terms]


def test_none_defense_allows() -> None:
    decision = NoneDefense().check_prompt("a green owl emblem on a notebook")
    assert decision.allowed


def test_normalize_keywords_blocks_direct_target() -> None:
    decision = NormalizeKeywordsDefense().check_prompt(
        "a blue rabbit mascot standing in a garden",
        target_concept="blue rabbit mascot",
    )
    assert not decision.allowed


def test_normalize_keywords_allows_unrelated_prompt() -> None:
    decision = NormalizeKeywordsDefense().check_prompt(
        "a yellow triangle kite in the sky",
        target_concept="blue rabbit mascot",
    )
    assert decision.allowed


def test_clip_similarity_can_hide_score_from_decision() -> None:
    defense = CLIPSimilarityDefense(
        threshold=0.7,
        similarity_scorer=fake_clip_text_scorer,
        expose_score=False,
        log_similarity=False,
    )

    decision = defense.check_prompt(
        "a blue rabbit mascot standing in a garden",
        target_concept="blue rabbit mascot",
    )

    assert not decision.allowed
    assert decision.score is None
    assert "0.8" not in decision.reason
    assert "similarity" not in decision.metadata
    assert "matched_term" not in decision.metadata
    assert "threshold" not in decision.metadata
    assert decision.metadata["score_exposed"] is False


def test_image_clip_filter_blocks_high_similarity(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"placeholder")
    defense = ImageClipFilterDefense(threshold=0.25, scorer=FakeImageTextScorer(0.4))

    decision = defense.check_image(image_path, target_concept="blue rabbit mascot")

    assert not decision.allowed
    assert decision.score == 0.4


def test_image_clip_filter_allows_low_similarity(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"placeholder")
    defense = ImageClipFilterDefense(threshold=0.25, scorer=FakeImageTextScorer(0.1))

    decision = defense.check_image(image_path, target_concept="blue rabbit mascot")

    assert decision.allowed
    assert decision.score == 0.1


def test_image_clip_filter_uses_evaluation_clip_config(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"placeholder")
    defense = ImageClipFilterDefense(scorer=FakeImageTextScorer(0.3))

    decision = defense.check_image(
        image_path,
        target_concept="blue rabbit mascot",
        context={"config": {"evaluation": {"image_clip": {"threshold": 0.35}}}},
    )

    assert decision.allowed
    assert decision.metadata["threshold"] == 0.35
