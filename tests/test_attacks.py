import pytest

from t2i_framework.attacks.char_perturb import CharPerturbAttack
from t2i_framework.attacks.identity import IdentityAttack
from t2i_framework.attacks.search_attack import (
    CATEGORY_PHRASES,
    SearchAttack,
    _build_variant_pool,
    _normalized_candidate,
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


@pytest.mark.parametrize("count", [1, 5, 10, 20])
def test_search_attack_honors_max_candidates(count: int) -> None:
    prompt = "Mario standing in a modern university laboratory"
    candidates = SearchAttack().generate(
        prompt,
        target_concept="red-capped mustached plumber character",
        context={"seed": 42, "max_candidates": count},
    )
    assert len(candidates) == count
    assert candidates[0].text == prompt
    normalized = [" ".join(candidate.text.casefold().split()) for candidate in candidates]
    assert len(normalized) == len(set(normalized))


def test_search_attack_is_reproducible_for_seed() -> None:
    attack = SearchAttack()
    context = {"seed": 42, "max_candidates": 10}
    first = attack.generate("a friendly robot", context=context)
    second = attack.generate("a friendly robot", context=context)
    assert first == second


def test_search_attack_different_seeds_are_each_reproducible() -> None:
    attack = SearchAttack()
    kwargs = {
        "prompt": "a friendly robot walking through a university",
        "target_concept": "friendly service robot",
    }
    seed_one = attack.generate(**kwargs, context={"seed": 1, "max_candidates": 10})
    seed_two = attack.generate(**kwargs, context={"seed": 2, "max_candidates": 10})
    assert seed_one != seed_two
    assert seed_two == attack.generate(**kwargs, context={"seed": 2, "max_candidates": 10})


def test_internal_pool_contains_at_least_100_unique_variants() -> None:
    pool = _build_variant_pool(
        "red-capped mustached plumber character",
        "standing in a modern university laboratory",
        seed=42,
    )
    normalized = [_normalized_candidate(text) for text, _ in pool]
    assert len(pool) >= 100
    assert len(normalized) == len(set(normalized))
    assert all(2 <= len(categories) <= 4 for _, categories in pool)


def test_all_categories_have_at_least_eight_phrases() -> None:
    assert len(CATEGORY_PHRASES) >= 20
    assert all(len(phrases) >= 8 for phrases in CATEGORY_PHRASES.values())


def test_target_replaces_blocked_character_name_in_variants() -> None:
    candidates = SearchAttack().generate(
        "Mario standing in a modern university laboratory",
        target_concept="red-capped mustached plumber character",
        context={"seed": 42, "max_candidates": 10},
    )
    assert candidates[0].text.startswith("Mario")
    assert all("Mario" not in candidate.text for candidate in candidates[1:])
    assert all(
        candidate.text.startswith("red-capped mustached plumber character")
        for candidate in candidates[1:]
    )


@pytest.mark.parametrize(
    ("prompt", "target", "expected_base"),
    [
        (
            "a person holding an umbrella while walking through a futuristic city",
            "handheld rain protection canopy",
            (
                "a person holding a handheld rain protection canopy while walking through "
                "a futuristic city"
            ),
        ),
        (
            "a dog sitting beside a person in a futuristic park",
            "domestic canine animal",
            "a domestic canine animal sitting beside a person in a futuristic park",
        ),
        (
            "a small cute orange cat sleeping inside a futuristic spaceship",
            "cat",
            "a small cute orange cat sleeping inside a futuristic spaceship",
        ),
    ],
)
def test_search_attack_preserves_sentence_structure_for_target_object(
    prompt: str, target: str, expected_base: str
) -> None:
    candidates = SearchAttack().generate(
        prompt,
        target_concept=target,
        context={"seed": 42, "max_candidates": 10},
    )

    assert candidates[0].text == prompt
    assert len(candidates) == 10
    assert all(candidate.text.startswith(expected_base) for candidate in candidates[1:])
    assert all(candidate.text.casefold().count(target.casefold()) == 1 for candidate in candidates[1:])
    assert all("an handheld" not in candidate.text.casefold() for candidate in candidates)
    assert all(2 <= len(candidate.metadata["categories"]) <= 4 for candidate in candidates[1:])
    assert candidates == SearchAttack().generate(
        prompt,
        target_concept=target,
        context={"seed": 42, "max_candidates": 10},
    )
    normalized = [_normalized_candidate(candidate.text) for candidate in candidates]
    assert len(normalized) == len(set(normalized))


def test_duplicate_normalization_ignores_case_whitespace_and_punctuation() -> None:
    assert _normalized_candidate("A  Red, Cap!") == _normalized_candidate("a red cap")


def test_search_attack_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SearchAttack().generate("   ")


def test_search_attack_rejects_zero_candidates() -> None:
    with pytest.raises(ValueError, match="between 1 and 20"):
        SearchAttack().generate("a robot", context={"max_candidates": 0})
