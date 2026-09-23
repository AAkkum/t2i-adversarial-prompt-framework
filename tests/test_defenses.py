from pathlib import Path

import pytest

from t2i_framework.defenses.clip_similarity import CLIPSimilarityDefense
from t2i_framework.defenses.character_filter import CharacterFilterDefense
from t2i_framework.defenses.semantic_concepts import SemanticMatch
from t2i_framework.defenses.image_clip_filter import ImageClipFilterDefense
from t2i_framework.defenses.latent_guard_lite import (
    LatentGuardLiteDefense,
    _LatentGuardLiteScorer,
)
from t2i_framework.defenses.none import NoneDefense
from t2i_framework.defenses.normalize_keywords import NormalizeKeywordsDefense


class FakeImageTextScorer:
    def __init__(self, score: float) -> None:
        self.fixed_score = score

    def score(self, image_path: Path, text: str) -> float:
        return self.fixed_score


def fake_clip_text_scorer(_prompt: str, terms: list[str]) -> list[float]:
    return [0.8 if term == "blue rabbit mascot" else 0.1 for term in terms]


class FakeLatentGuardScorer:
    def __init__(self) -> None:
        self.seen_terms: list[str] = []

    def __call__(self, _prompt: str, terms: list[str]) -> list[float]:
        self.seen_terms = list(terms)
        return [9.5 if term == "mario" else 1.0 for term in terms]


class FakeCharacterMatcher:
    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.seen_concepts: list[dict[str, str]] = []

    def match(self, _text: str, concepts: dict[str, str]) -> SemanticMatch:
        self.seen_concepts.append(dict(concepts))
        term = next(iter(concepts))
        return SemanticMatch(term, concepts[term], self.score)


class FakeCaptioner:
    def __init__(self, caption: str = "target-like generated caption") -> None:
        self.caption_text = caption

    def caption(self, _image_path: Path) -> str:
        return self.caption_text


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


def test_latent_guard_lite_blocks_runtime_target() -> None:
    scorer = FakeLatentGuardScorer()
    defense = LatentGuardLiteDefense(
        threshold=9.0131,
        scorer=scorer,
        expose_score=True,
        log_score=False,
    )

    decision = defense.check_prompt("red capped hero jumping", target_concept="mario")

    assert not decision.allowed
    assert decision.score == 9.5
    assert decision.metadata["matched_term"] == "mario"
    assert "mario" in scorer.seen_terms


def test_latent_guard_lite_can_hide_score_from_decision() -> None:
    defense = LatentGuardLiteDefense(
        threshold=9.0131,
        scorer=FakeLatentGuardScorer(),
        expose_score=False,
        log_score=False,
    )

    decision = defense.check_prompt("red capped hero jumping", target_concept="mario")

    assert not decision.allowed
    assert decision.score is None
    assert "9.5" not in decision.reason
    assert "matched_term" not in decision.metadata
    assert decision.metadata["score_exposed"] is False


def test_latent_guard_lite_uses_context_config(tmp_path: Path) -> None:
    concepts_path = tmp_path / "concepts.yaml"
    weights_path = tmp_path / "model_parameters.pth"
    concepts_path.write_text(
        """
concepts:
  blue rabbit mascot:
    aliases:
      - blue bunny mascot
""".strip(),
        encoding="utf-8",
    )
    weights_path.write_bytes(b"placeholder")
    defense = LatentGuardLiteDefense(scorer=lambda _prompt, terms: [1.0 for _ in terms])

    decision = defense.check_prompt(
        "a harmless prompt",
        context={
            "config": {
                "defense": {
                    "concepts_path": str(concepts_path),
                    "weights_path": str(weights_path),
                    "threshold": 2.0,
                    "include_aliases": False,
                    "use_target_concept": False,
                    "fail_on_error": True,
                    "expose_score": True,
                    "log_score": False,
                }
            }
        },
    )

    assert decision.allowed
    assert defense.concepts_path == concepts_path
    assert defense.weights_path == weights_path
    assert defense.threshold == 2.0
    assert defense.include_aliases is False
    assert defense.use_target_concept is False
    assert defense.fail_on_error is True
    assert decision.metadata["checked_terms"] == 1


def test_latent_guard_lite_raises_when_strict_mode_cannot_load_weights(
    tmp_path: Path,
) -> None:
    defense = LatentGuardLiteDefense(
        weights_path=tmp_path / "missing.pth",
        fail_on_error=True,
        log_score=False,
    )

    with pytest.raises(RuntimeError, match="weights not found"):
        defense.check_prompt("red capped hero jumping", target_concept="mario")


def test_latent_guard_lite_keeps_loaded_scorer_when_context_is_unchanged() -> None:
    defense = LatentGuardLiteDefense(
        scorer=lambda _prompt, terms: [1.0 for _ in terms],
        log_score=False,
    )
    loaded_scorer = object()
    defense._latent_scorer = loaded_scorer

    defense.check_prompt(
        "red capped hero jumping",
        target_concept="mario",
        context={
            "config": {
                "defense": {
                    "model_id": defense.model_id,
                    "device": defense.device,
                    "num_heads": defense.num_heads,
                    "head_dim": defense.head_dim,
                    "out_dim": defense.out_dim,
                    "input_dim": defense.input_dim,
                    "batch_size": defense.batch_size,
                }
            }
        },
    )

    assert defense._latent_scorer is loaded_scorer


def test_latent_guard_lite_invalidates_loaded_scorer_when_model_config_changes() -> None:
    defense = LatentGuardLiteDefense(
        scorer=lambda _prompt, terms: [1.0 for _ in terms],
        log_score=False,
    )
    defense._latent_scorer = object()

    defense.check_prompt(
        "red capped hero jumping",
        target_concept="mario",
        context={"config": {"defense": {"batch_size": defense.batch_size + 1}}},
    )

    assert defense._latent_scorer is None


def test_latent_guard_lite_caches_concept_embeddings(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    scorer = _LatentGuardLiteScorer(
        weights_path=tmp_path / "unused.pth",
        model_id="unused",
        device="cpu",
        num_heads=16,
        head_dim=32,
        out_dim=128,
        input_dim=768,
        batch_size=64,
    )
    scorer._torch = torch
    embedded_batches: list[list[str]] = []

    def fake_embed_texts(texts: list[str]):
        embedded_batches.append(list(texts))
        return torch.ones((len(texts), 2, 3))

    scorer._embed_texts = fake_embed_texts

    scorer._concept_embeddings(["mario", "adidas"])
    scorer._concept_embeddings(["adidas", "mario", "butterfly"])

    assert embedded_batches == [["mario", "adidas"], ["butterfly"]]
    assert set(scorer._concept_embedding_cache) == {"mario", "adidas", "butterfly"}


def test_character_filter_blocks_target_as_runtime_keyword(tmp_path: Path) -> None:
    terms_path = tmp_path / "blocked_terms.txt"
    concepts_path = tmp_path / "concept_targets.json"
    terms_path.write_text("", encoding="utf-8")
    concepts_path.write_text("{}", encoding="utf-8")
    defense = CharacterFilterDefense(
        terms_path=terms_path,
        concepts_path=concepts_path,
        enable_semantic_prompt=False,
        enable_image_semantic=False,
    )

    decision = defense.check_prompt("mario jumping in the air", target_concept="mario")

    assert not decision.allowed
    assert decision.metadata["matched_keyword_block"] == "mario"
    assert decision.metadata["closest_protected_concept"] == "mario"
    assert decision.metadata["target_concept_used"] is True


def test_character_filter_adds_target_to_semantic_prompt_concepts(tmp_path: Path) -> None:
    terms_path = tmp_path / "blocked_terms.txt"
    concepts_path = tmp_path / "concept_targets.json"
    terms_path.write_text("", encoding="utf-8")
    concepts_path.write_text("{}", encoding="utf-8")
    matcher = FakeCharacterMatcher(score=0.9)
    defense = CharacterFilterDefense(
        terms_path=terms_path,
        concepts_path=concepts_path,
        semantic_matcher=matcher,
        captioner=FakeCaptioner(),
        enable_image_semantic=False,
    )

    decision = defense.check_prompt("red capped hero in a platform game", target_concept="mario")

    assert not decision.allowed
    assert matcher.seen_concepts[0]["mario"] == "mario"
    assert decision.metadata["target_concept_used"] is True


def test_character_filter_adds_target_to_image_semantic_concepts(tmp_path: Path) -> None:
    terms_path = tmp_path / "blocked_terms.txt"
    concepts_path = tmp_path / "concept_targets.json"
    image_path = tmp_path / "image.png"
    terms_path.write_text("", encoding="utf-8")
    concepts_path.write_text("{}", encoding="utf-8")
    image_path.write_bytes(b"placeholder")
    matcher = FakeCharacterMatcher(score=0.9)
    defense = CharacterFilterDefense(
        terms_path=terms_path,
        concepts_path=concepts_path,
        semantic_matcher=matcher,
        captioner=FakeCaptioner("red capped hero in a platform game"),
    )

    decision = defense.check_image(image_path, target_concept="mario")

    assert not decision.allowed
    assert matcher.seen_concepts[0]["mario"] == "mario"
    assert decision.metadata["target_concept_used"] is True


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
