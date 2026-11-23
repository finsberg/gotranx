import re
from pathlib import Path

import gotranx
from gotranx.cli import app
from typer.testing import CliRunner

FILES = Path(__file__).parent / "compose_files"
runner = CliRunner()


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_cli_compose_writes_and_reports(tmp_path):
    out = tmp_path / "m.ode"
    result = runner.invoke(app, ["compose", str(FILES / "toy_recipe.toml"), "-o", str(out)])
    assert result.exit_code == 0, result.output
    text = strip_ansi(result.stdout)
    assert "ep needs: J" in text
    assert "toy needs: cai" in text
    assert "Warning" not in text
    gotranx.load_ode(out)


def test_cli_compose_default_output_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["compose", str(FILES / "toy_recipe.toml")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "toy_recipe.ode").is_file()


def test_cli_compose_error_exit_code(tmp_path):
    (tmp_path / "bad.glue.ode").write_text('expressions("ep")\nreplace nothing_here = 1.0\n')
    recipe = tmp_path / "bad.toml"
    recipe.write_text(
        'time_unit = "ms"\nglue = "bad.glue.ode"\n'
        f'[[parts]]\nname = "ep"\nfile = "{FILES / "toy_ep.ode"}"\ntime_unit = "ms"\n'
    )
    out = tmp_path / "bad.ode"
    result = runner.invoke(app, ["compose", str(recipe), "-o", str(out)])
    assert result.exit_code != 0
    assert "nothing_here is not a name of ep" in strip_ansi(result.output)
    assert not out.exists()


def test_cli_compose_warns_for_allow_missing(tmp_path):
    (tmp_path / "g.ode").write_text('expressions("ep")\nextra = outside*2.0\n')
    recipe = tmp_path / "r.toml"
    recipe.write_text(
        'time_unit = "ms"\nglue = "g.ode"\nallow_missing = true\n'
        f'[[parts]]\nname = "ep"\nfile = "{FILES / "toy_ep.ode"}"\ntime_unit = "ms"\n'
    )
    out = tmp_path / "r.ode"
    result = runner.invoke(app, ["compose", str(recipe), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "Warning" in strip_ansi(result.stdout) and "outside" in strip_ansi(result.stdout)
