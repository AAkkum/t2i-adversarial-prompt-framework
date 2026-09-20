from t2i_framework.paraphrasers.qwen_ollama import (
    QwenOllamaParaphraser,
    infer_phrase_role,
    normalize_detail_level,
)
from t2i_framework.paraphrasers.qwen_transformers import QwenTransformersParaphraser


class FakeTransformersBackend:
    model_id = "Qwen/test"

    def generate(self, prompt, **kwargs):
        return '["first option", "second option"]'

    def unload(self):
        return None


def test_infer_phrase_role_detects_action_phrase() -> None:
    assert infer_phrase_role("surrendering") == "action or verb phrase"
    assert infer_phrase_role("playing golf") == "action or verb phrase"


def test_infer_phrase_role_defaults_to_noun_phrase() -> None:
    assert infer_phrase_role("black cat") == "noun phrase"


def test_qwen_prompt_preserves_action_phrase_role() -> None:
    prompt = QwenOllamaParaphraser()._build_prompt("surrendering", "a figure surrendering", 5)  # noqa: SLF001

    assert "action or verb phrase" in prompt
    assert "return only action or verb phrases" in prompt


def test_qwen_prompt_preserves_noun_phrase_role() -> None:
    prompt = QwenOllamaParaphraser()._build_prompt("black cat", "the black cat was hunting a rat", 5)  # noqa: SLF001

    assert "noun phrase" in prompt
    assert "return only noun phrases" in prompt


def test_qwen_prompt_uses_medium_detail_for_noun_phrases() -> None:
    prompt = QwenOllamaParaphraser(detail_level="medium")._build_prompt(
        "mario",
        "mario jumping",
        5,
    )  # noqa: SLF001

    assert "Detail level:" in prompt
    assert "`medium`" in prompt
    assert "Use about 8-18 words" in prompt
    assert "2-4 distinctive visual attributes" in prompt


def test_qwen_prompt_keeps_actions_compact_even_when_detailed() -> None:
    prompt = QwenOllamaParaphraser(detail_level="detailed")._build_prompt(
        "surrendering",
        "a figure surrendering",
        5,
    )  # noqa: SLF001

    assert "`detailed`" in prompt
    assert "Ignore medium/detailed noun-description behavior" in prompt
    assert "about 1-6 words" in prompt


def test_qwen_detail_level_aliases() -> None:
    assert normalize_detail_level("balanced") == "medium"
    assert normalize_detail_level("long") == "detailed"


def test_qwen_paraphraser_stores_raw_response() -> None:
    paraphraser = QwenOllamaParaphraser()
    paraphraser._generate = lambda prompt: '["first option", "second option"]'  # noqa: SLF001

    candidates = paraphraser.generate_candidates("source phrase", count=5)

    assert candidates == ["first option", "second option"]
    assert paraphraser.last_raw_response == '["first option", "second option"]'
    assert paraphraser.last_candidates == ["first option", "second option"]


def test_transformers_paraphraser_reuses_qwen_prompt_and_parser() -> None:
    paraphraser = QwenTransformersParaphraser(FakeTransformersBackend())

    candidates = paraphraser.generate_candidates("source phrase", count=5)

    assert candidates == ["first option", "second option"]
    assert paraphraser.last_raw_response == '["first option", "second option"]'
