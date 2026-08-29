from pathlib import Path

from typer.testing import CliRunner

from main import app


def test_cli_mock_creates_result_and_image(tmp_path: Path) -> None:
    out = tmp_path / "run"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--model",
            "mock",
            "--attack",
            "identity",
            "--defense",
            "none",
            "--prompt",
            "a blue rabbit mascot standing in a garden",
            "--target",
            "blue rabbit mascot",
            "--seed",
            "42",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "results.jsonl").exists()
    images = list((out / "images").glob("*.png"))
    assert len(images) == 1


def test_cli_mock_can_write_multiple_attack_candidates(tmp_path: Path) -> None:
    out = tmp_path / "run"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--model",
            "mock",
            "--attack",
            "groot_lite",
            "--defense",
            "none",
            "--prompt",
            "a blue rabbit mascot standing in a garden",
            "--target",
            "blue rabbit mascot",
            "--max-candidates",
            "3",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len((out / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    images = list((out / "images").glob("*.png"))
    assert len(images) == 3
