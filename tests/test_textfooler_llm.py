from t2i_framework.attacks.textfooler_llm import (
    build_paraphrase_prompt,
    infer_phrase_role,
    normalize_detail_level,
    parse_candidate_list,
)


def test_infer_phrase_role_detects_action_phrase() -> None:
    assert infer_phrase_role("surrendering") == "action or verb phrase"
    assert infer_phrase_role("playing golf") == "action or verb phrase"


def test_infer_phrase_role_defaults_to_noun_phrase() -> None:
    assert infer_phrase_role("black cat") == "noun phrase"


def test_local_llm_prompt_preserves_action_phrase_role() -> None:
    prompt = build_paraphrase_prompt("surrendering", "a figure surrendering", 5)

    assert "action or verb phrase" in prompt
    assert "return only action or verb phrases" in prompt


def test_local_llm_prompt_preserves_existing_action_object() -> None:
    prompt = build_paraphrase_prompt(
        "chasing a balloon",
        "a dog chasing a balloon through a park",
        5,
    )

    assert "Preserve every object" in prompt
    assert "`chasing a balloon` -> `pursuing a balloon`" in prompt
    assert "every object or complement already contained" in prompt
    assert "without subject, object" not in prompt


def test_local_llm_prompt_preserves_noun_phrase_role() -> None:
    prompt = build_paraphrase_prompt("black cat", "the black cat was hunting a rat", 5)

    assert "noun phrase" in prompt
    assert "return only noun phrases" in prompt


def test_local_llm_prompt_uses_medium_detail_for_noun_phrases() -> None:
    prompt = build_paraphrase_prompt(
        "mario",
        "mario jumping",
        5,
        detail_level="medium",
    )

    assert "Detail level:" in prompt
    assert "`medium`" in prompt
    assert "Use about 8-18 words" in prompt
    assert "2-4 distinctive visual attributes" in prompt


def test_local_llm_prompt_keeps_actions_compact_even_when_detailed() -> None:
    prompt = build_paraphrase_prompt(
        "surrendering",
        "a figure surrendering",
        5,
        detail_level="detailed",
    )

    assert "`detailed`" in prompt
    assert "Ignore medium/detailed noun-description behavior" in prompt
    assert "about 1-10 words" in prompt


def test_local_llm_detail_level_aliases() -> None:
    assert normalize_detail_level("balanced") == "medium"
    assert normalize_detail_level("long") == "detailed"


def test_local_llm_candidate_parser_accepts_json_array() -> None:
    candidates = parse_candidate_list('["first option", "second option"]', limit=5)
    assert candidates == ["first option", "second option"]
