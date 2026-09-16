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


def test_textfooler_similarity_config_can_disable_similarity() -> None:
    attack = TextFoolerStyleAttack(
        paraphraser=lambda _unit, _context, _count: ["red plumber"],
        candidate_count=1,
        max_rounds=1,
        log_similarity=False,
        unload_ollama_after_attack=False,
    )

    candidates = attack.generate(
        "mario jumping",
        target_concept="mario",
        context={
            "defense": TargetBlockingDefense(),
            "config": {"attack": {"similarity": {"enabled": False}}},
        },
    )

    assert candidates[0].text == "red plumber jumping"
    assert candidates[0].metadata["selected_similarity"] == 1.0
    assert candidates[0].metadata["selected_similarity_method"] == "none"


def test_textfooler_similarity_config_accepts_sentence_transformer_alias() -> None:
    attack = TextFoolerStyleAttack(use_clip_similarity=False)

    attack._apply_context_config(
        {
            "config": {
                "attack": {
                    "similarity": {
                        "enabled": True,
                        "method": "minilm",
                        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                        "device": "cpu",
                        "threshold": 0.62,
                    }
                }
            }
        }
    )

    assert attack.similarity_method == "sentence_transformer"
    assert attack.similarity_model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert attack.similarity_device == "cpu"
    assert attack.min_similarity == 0.62


def test_textfooler_accepts_grouped_paraphraser_judge_and_search_config() -> None:
    attack = TextFoolerStyleAttack()

    attack._apply_context_config(
        {
            "config": {
                "attack": {
                    "search": {
                        "candidate_count": 3,
                        "max_rounds": 4,
                        "max_candidate_batches": 5,
                        "filter_context_leaks": True,
                        "log_candidate_filtering": True,
                    },
                    "paraphraser": {
                        "enabled": False,
                        "model": "qwen3:8b",
                        "detail_level": "detailed",
                        "log_raw": True,
                        "unload_after_attack": False,
                        "log_unload": False,
                    },
                    "judge": {
                        "enabled": True,
                        "model": "qwen3:14b",
                        "threshold": 0.81,
                        "log": False,
                    },
                }
            }
        }
    )

    assert attack.candidate_count == 3
    assert attack.max_rounds == 4
    assert attack.max_candidate_batches == 5
    assert attack.filter_context_leaks is True
    assert attack.log_candidate_filtering is True
    assert attack.use_qwen_fallback is False
    assert attack.paraphraser_model == "qwen3:8b"
    assert attack.paraphraser_detail_level == "detailed"
    assert attack.log_raw_paraphrases is True
    assert attack.unload_ollama_after_attack is False
    assert attack.log_ollama_unload is False
    assert attack.use_llm_judge_fallback is True
    assert attack.llm_judge_model == "qwen3:14b"
    assert attack.llm_judge_threshold == 0.81
    assert attack.log_llm_judge is False
