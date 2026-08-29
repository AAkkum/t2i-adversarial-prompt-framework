from pathlib import Path

from t2i_framework.attacks.char_perturb import CharPerturbAttack
from t2i_framework.attacks.groot_lite import GrootLiteAttack
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
