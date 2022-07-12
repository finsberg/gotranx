import gotran_parser
from typer.testing import CliRunner

runner = CliRunner(mix_stderr=False)


def test_cli_version():
    result = runner.invoke(gotran_parser.cli.app, "--version")
    assert result.exit_code == 0
    assert "gotran-parser" in result.stdout


def test_cli_license():
    result = runner.invoke(gotran_parser.cli.app, "--license")
    assert result.exit_code == 0
    assert "MIT" in result.stdout
