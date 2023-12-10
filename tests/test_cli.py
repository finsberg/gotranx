from __future__ import annotations
import gotranx
from typer.testing import CliRunner

runner = CliRunner(mix_stderr=False)


def test_cli_version():
    result = runner.invoke(gotranx.cli.app, "--version")
    assert result.exit_code == 0
    assert "gotranx" in result.stdout


def test_cli_license():
    result = runner.invoke(gotranx.cli.app, "--license")
    assert result.exit_code == 0
    assert "MIT" in result.stdout
