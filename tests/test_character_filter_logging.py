import json
import tempfile
from pathlib import Path

from t2i_framework.core.logging_utils import console
from t2i_framework.defenses.character_filter import CharacterFilterDefense
from t2i_framework.defenses.semantic_concepts import SemanticMatch


class _Matcher:
    def match(self, _text: str, concepts: dict[str, str]) -> SemanticMatch:
        term = next(iter(concepts))
        return SemanticMatch(term, concepts[term], 0.75)


class _Captioner:
    def caption(self, _image_path: Path) -> str:
        return "a red-capped platform hero"


def _defense_files(directory: Path) -> tuple[Path, Path]:
    terms_path = directory / "blocked_terms.txt"
    concepts_path = directory / "concept_targets.json"
    terms_path.write_text("mario\n", encoding="utf-8")
    concepts_path.write_text(
        json.dumps({"mario": "red-capped platform hero"}),
        encoding="utf-8",
    )
    return terms_path, concepts_path


def test_character_filter_prompt_log_is_compact() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        terms_path, concepts_path = _defense_files(Path(temp_dir))
        defense = CharacterFilterDefense(
            terms_path=terms_path,
            concepts_path=concepts_path,
            enable_semantic_prompt=False,
            enable_image_semantic=False,
        )

        with console.capture() as captured:
            defense.check_prompt(
                "mario jumping",
                target_concept="mario",
                context={"candidate_index": 2},
            )

    output = captured.get()
    assert output.count("[character_filter]") == 1
    assert "stage=prompt candidate=2 method=keyword" in output
    assert 'matched="mario"' in output
    assert "allowed=False blocked_by=KEYWORD" in output
    assert "Defense result for Candidate" not in output


def test_character_filter_image_logs_caption_and_similarity() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        terms_path, concepts_path = _defense_files(directory)
        image_path = directory / "image.png"
        image_path.write_bytes(b"placeholder")
        defense = CharacterFilterDefense(
            terms_path=terms_path,
            concepts_path=concepts_path,
            semantic_matcher=_Matcher(),
            captioner=_Captioner(),
            semantic_image_threshold=0.5,
        )

        with console.capture() as captured:
            defense.check_image(
                image_path,
                target_concept="mario",
                context={"candidate_index": 3},
            )

    output = captured.get()
    assert output.count("[character_filter]") == 2
    assert "stage=image_caption candidate=3" in output
    assert 'caption="a red-capped platform hero"' in output
    assert "stage=image candidate=3 method=blip_minilm" in output
    assert "score=0.750 threshold=0.500 allowed=False" in output


def test_character_filter_prompt_similarity_log_matches_textfooler_style() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        terms_path, concepts_path = _defense_files(Path(temp_dir))
        defense = CharacterFilterDefense(
            terms_path=terms_path,
            concepts_path=concepts_path,
            semantic_matcher=_Matcher(),
            semantic_prompt_threshold=0.5,
            enable_image_semantic=False,
        )

        with console.capture() as captured:
            defense.check_prompt(
                "a red-capped platform hero jumping",
                target_concept="mario",
                context={"candidate_index": 4},
            )

    output = captured.get()
    assert output.count("[character_filter]") == 1
    assert "stage=prompt candidate=4 method=minilm" in output
    assert 'matched="mario"' in output
    assert 'concept="red-capped platform hero"' in output
    assert "score=0.750 threshold=0.500 allowed=False" in output
