from t2i_framework.attacks.char_perturb import CharPerturbAttack
from t2i_framework.attacks.identity import IdentityAttack


def test_identity_returns_original_prompt() -> None:
    prompt = "a blue rabbit mascot standing in a garden"
    candidates = IdentityAttack().generate(prompt)
    assert candidates[0].text == prompt


def test_char_perturb_returns_candidates() -> None:
    prompt = "a red cube robot holding a balloon"
    candidates = CharPerturbAttack().generate(prompt, context={"seed": 42})
    assert len(candidates) >= 1
    assert all(candidate.text for candidate in candidates)
