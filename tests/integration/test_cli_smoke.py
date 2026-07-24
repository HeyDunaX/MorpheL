"""Integration test for Typer CLI."""

from typer.testing import CliRunner

from morphel.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "MorpheL" in result.stdout


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "MorpheL version" in result.stdout


def test_cli_dry_run_validation(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "train-tokenizer",
            "--language", "tr",
            "--vowels", "aeiou",
            "--output-dir", str(tmp_path / "out"),
            "--splits", "train",
            "--dry-run"
        ],
        catch_exceptions=False,
    )
    print("CLI Output:", result.output)
    assert result.exit_code == 0
    assert "Dry run successful" in result.output
