import pytest

from t2i_framework.defenses.character_filter import CharacterFilterDefense
from t2i_framework.defenses.filter_placeholder import FilterPlaceholderDefense
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


def test_filter_placeholder_returns_explicit_allow_decision() -> None:
    decision = FilterPlaceholderDefense().check_prompt("a harmless blue robot")
    assert decision.allowed is True
    assert decision.reason == "Placeholder filter: no blocking rule configured"


@pytest.mark.parametrize(
    ("prompt", "matched"),
    [
        ("Mario in a laboratory", "mario"),
        ("MARIO in a laboratory", "mario"),
        ("Super Mario in a laboratory", "super mario"),
        ("A Nintendo character", "nintendo"),
        ("A scene in the Mushroom Kingdom", "mushroom kingdom"),
    ],
)
def test_character_filter_blocks_terms(prompt: str, matched: str) -> None:
    decision = CharacterFilterDefense().check_prompt(prompt)
    assert not decision.allowed
    assert matched in decision.reason


def test_character_filter_allows_neutral_prompt() -> None:
    decision = CharacterFilterDefense().check_prompt(
        "a cheerful mustached plumber wearing a red cap and blue overalls"
    )
    assert decision.allowed
