from pathlib import Path

import pytest

from t2i_framework.defenses.latent_guard_lite import (
    LatentGuardLiteDefense,
    _LatentGuardLiteScorer,
)
from t2i_framework.defenses.none import NoneDefense


class FakeLatentGuardScorer:
    def __init__(self) -> None:
        self.seen_terms: list[str] = []

    def __call__(self, _prompt: str, terms: list[str]) -> list[float]:
        self.seen_terms = list(terms)
        return [9.5 if term == "mario" else 1.0 for term in terms]


def test_none_defense_allows() -> None:
    decision = NoneDefense().check_prompt("a green owl emblem on a notebook")
    assert decision.allowed


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


def test_latent_guard_lite_invalidates_scorer_when_model_config_changes() -> None:
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
