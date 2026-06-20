from t2i_framework.defenses.none import NoneDefense
from t2i_framework.defenses.normalize_keywords import NormalizeKeywordsDefense


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
