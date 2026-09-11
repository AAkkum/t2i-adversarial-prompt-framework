from t2i_framework.paraphrasers.qwen_ollama import QwenOllamaParaphraser, infer_phrase_role


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


def test_qwen_paraphraser_stores_raw_response() -> None:
    paraphraser = QwenOllamaParaphraser()
    paraphraser._generate = lambda prompt: '["first option", "second option"]'  # noqa: SLF001

    candidates = paraphraser.generate_candidates("source phrase", count=5)

    assert candidates == ["first option", "second option"]
    assert paraphraser.last_raw_response == '["first option", "second option"]'
    assert paraphraser.last_candidates == ["first option", "second option"]
