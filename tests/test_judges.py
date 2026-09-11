import pytest

from t2i_framework.judges.ollama_similarity import parse_score


def test_parse_score_reads_json_score() -> None:
    assert parse_score('{"score": 0.83}') == 0.83


def test_parse_score_reads_json_inside_text() -> None:
    assert parse_score('Result: {"score": "0.76"}') == 0.76


def test_parse_score_clamps_score() -> None:
    assert parse_score('{"score": 1.2}') == 1.0


def test_parse_score_raises_for_missing_score() -> None:
    with pytest.raises(RuntimeError):
        parse_score("not a score")
