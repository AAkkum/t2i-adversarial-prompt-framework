from pathlib import Path

from t2i_framework.attacks.char_perturb import CharPerturbAttack
from t2i_framework.attacks.groot_lite import GrootLiteAttack
from t2i_framework.attacks.identity import IdentityAttack
from t2i_framework.attacks.textfooler_style import TextFoolerStyleAttack
from t2i_framework.core.types import DefenseDecision


class TargetBlockingDefense:
    def check_prompt(self, prompt: str, target_concept=None, context=None) -> DefenseDecision:
        blocked = bool(target_concept and target_concept.lower() in prompt.lower())
        return DefenseDecision(
            allowed=not blocked,
            reason="blocked target" if blocked else "allowed",
            score=1.0 if blocked else 0.0,
            metadata={},
        )


def test_identity_returns_original_prompt() -> None:
    prompt = "a blue rabbit mascot standing in a garden"
    candidates = IdentityAttack().generate(prompt)
    assert candidates[0].text == prompt


def test_char_perturb_returns_candidates() -> None:
    prompt = "a red cube robot holding a balloon"
    candidates = CharPerturbAttack().generate(prompt, context={"seed": 42})
    assert len(candidates) >= 1
    assert all(candidate.text for candidate in candidates)


def test_groot_lite_decomposes_known_target() -> None:
    prompt = "a blue rabbit mascot standing in a garden"
    candidates = GrootLiteAttack().generate(prompt, target_concept="blue rabbit mascot")

    assert len(candidates) >= 1
    assert candidates[0].text != prompt
    assert "blue rabbit mascot" not in candidates[0].text.lower()
    assert candidates[0].metadata["strategy"] == "semantic_decomposition"


def test_groot_lite_falls_back_for_unknown_target() -> None:
    prompt = "a yellow triangle kite in the sky"
    candidates = GrootLiteAttack().generate(prompt, target_concept="yellow triangle kite")

    assert candidates[0].text == prompt
    assert candidates[0].metadata["status"] == "unsupported_target_concept"


def test_groot_lite_loads_decompositions_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "decompositions.yaml"
    path.write_text(
        """
concepts:
  yellow triangle kite:
    decompositions:
      - yellow three-cornered flying toy
""".strip(),
        encoding="utf-8",
    )

    candidates = GrootLiteAttack(decompositions_path=path).generate(
        "a yellow triangle kite in the sky",
        target_concept="yellow triangle kite",
    )

    assert candidates[0].text == "a yellow three-cornered flying toy in the sky"


def test_groot_lite_uses_decomposition_path_from_context(tmp_path: Path) -> None:
    path = tmp_path / "decompositions.yaml"
    path.write_text(
        """
concepts:
  yellow triangle kite:
    decompositions:
      - yellow three-cornered flying toy
""".strip(),
        encoding="utf-8",
    )

    candidates = GrootLiteAttack().generate(
        "a yellow triangle kite in the sky",
        target_concept="yellow triangle kite",
        context={"config": {"attack": {"decompositions_path": str(path)}}},
    )

    assert candidates[0].text == "a yellow three-cornered flying toy in the sky"


def test_textfooler_selects_highest_llm_judge_similarity() -> None:
    judge_scores = {
        "first visual replacement": 0.80,
        "better visual replacement": 0.95,
    }
    attack = TextFoolerStyleAttack(
        paraphraser=lambda _unit, _context, _count: [
            "first visual replacement",
            "better visual replacement",
        ],
        candidate_count=2,
        max_rounds=1,
        similarity_scorer=lambda _source, _candidate: 0.20,
        use_llm_judge_fallback=True,
        llm_judge=lambda _source, candidate, _context: judge_scores[candidate],
        llm_judge_threshold=0.75,
        log_similarity=False,
        log_llm_judge=False,
        unload_ollama_after_attack=False,
    )

    candidates = attack.generate(
        "mario jumping",
        target_concept="mario",
        context={"defense": TargetBlockingDefense()},
    )

    assert candidates[0].text == "better visual replacement jumping"
    assert candidates[0].metadata["selected_similarity"] == 0.95
    assert candidates[0].metadata["selected_similarity_method"] == "llm_judge"
