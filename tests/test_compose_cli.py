import re
import shutil
from pathlib import Path

import pytest
import sympy as sp

import gotranx
from gotranx.cli import app
from typer.testing import CliRunner

FILES = Path(__file__).parent / "compose_files"
runner = CliRunner()


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def needs_lines(text: str) -> set[str]:
    return {line for line in strip_ansi(text).splitlines() if " needs: " in line}


def test_cli_compose_writes_and_reports(tmp_path):
    out = tmp_path / "m.ode"
    result = runner.invoke(app, ["compose", str(FILES / "toy_recipe.toml"), "-o", str(out)])
    assert result.exit_code == 0, result.output
    text = strip_ansi(result.stdout)
    # One line per part, not per component (ep.calcium, ep.gate, ...)
    assert needs_lines(text) == {"ep needs: J", "toy needs: cai"}
    assert "Warning" not in text
    gotranx.load_ode(out)


def test_cli_compose_reports_a_split(tmp_path):
    recipe = toy_in(tmp_path)
    table = '\n[components.mechanics]\ninputs = ["toy.B"]\noutputs = ["toy.Zs", "toy.Ta"]\n'
    recipe.write_text(recipe.read_text() + table)
    result = runner.invoke(app, ["compose", str(recipe), "-o", str(tmp_path / "m.ode")])
    assert result.exit_code == 0, result.output
    assert needs_lines(result.stdout) == {
        "ep needs: J",
        "toy needs: cai",
        "mechanics needs: B",
    }


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
        f'[[parts]]\nname = "ep"\nfile = "{(FILES / "toy_ep.ode").as_posix()}"\ntime_unit = "ms"\n'
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
        f'[[parts]]\nname = "ep"\nfile = "{(FILES / "toy_ep.ode").as_posix()}"\ntime_unit = "ms"\n'
    )
    out = tmp_path / "r.ode"
    result = runner.invoke(app, ["compose", str(recipe), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "Warning" in strip_ansi(result.stdout) and "outside" in strip_ansi(result.stdout)


# A gate rate with a removable singularity at V = -10
SINGULAR = """
states("membrane", V = -80.0)
states("gate", m = 0.5)
expressions("membrane")
dV_dt = -0.01*(V + 80)
expressions("gate")
alpha = 0.1*(V + 10)/(exp((V + 10)/10) - 1)
dm_dt = alpha*(1 - m) - 0.1*m
"""


def nested_guards(ode: gotranx.ODE) -> list[str]:
    """The assignments with a Piecewise inside a Piecewise"""
    return [
        a.name
        for a in ode.intermediates + ode.state_derivatives
        if any(arg.has(sp.Piecewise) for p in a.expr.atoms(sp.Piecewise) for arg in p.args)
    ]


def test_cli_compose_writes_an_unguarded_model(tmp_path):
    # Like every other .ode file, so that loading it guards it once, as compose does
    (tmp_path / "g.ode").write_text(SINGULAR)
    recipe = tmp_path / "m.toml"
    recipe.write_text('time_unit = "ms"\n[[parts]]\nname = "g"\nfile = "g.ode"\ntime_unit = "ms"\n')
    out = tmp_path / "m.ode"
    result = runner.invoke(app, ["compose", str(recipe), "-o", str(out)])
    assert result.exit_code == 0, result.output
    unguarded = gotranx.load_ode(out, remove_singularities=False)
    assert not any(a.expr.has(sp.Piecewise) for a in unguarded.intermediates)
    assert unguarded == gotranx.compose(recipe, remove_singularities=False)

    guarded = gotranx.load_ode(out)
    composed = gotranx.compose(recipe)
    assert composed["alpha"].expr.has(sp.Piecewise)
    assert nested_guards(guarded) == []
    assert guarded == composed


def toy_in(folder: Path) -> Path:
    """The toy composition, copied into ``folder``, with the recipe named after its first
    part: ``toy_ep.toml``, whose default output is ``toy_ep.ode``"""
    shutil.copy(FILES / "toy_ep.ode", folder)
    shutil.copy(FILES / "toy_glue.ode", folder)
    shutil.copy(FILES.parent / "odefiles" / "contraction_toy.ode", folder)
    recipe = (FILES / "toy_recipe.toml").read_text()
    recipe = recipe.replace("../odefiles/contraction_toy.ode", "contraction_toy.ode")
    (folder / "toy_ep.toml").write_text(recipe)
    return folder / "toy_ep.toml"


def contents(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file()}


def test_cli_compose_default_output_is_not_a_part(tmp_path, monkeypatch):
    # The default output, toy_ep.ode in the current directory, is the part ep's file
    toy_in(tmp_path)
    monkeypatch.chdir(tmp_path)
    before = contents(tmp_path)
    result = runner.invoke(app, ["compose", "toy_ep.toml"])
    assert result.exit_code != 0
    assert strip_ansi(result.output).strip() == (
        "Not writing toy_ep.ode: it is the file of part ep. Choose another output with -o."
    )
    assert contents(tmp_path) == before


@pytest.mark.parametrize(
    "output, what",
    [
        ("toy_ep.toml", "the recipe"),
        ("toy_glue.ode", "the glue file"),
        ("contraction_toy.ode", "the file of part toy"),
        ("sub/../toy_ep.ode", "the file of part ep"),  # The same file, written otherwise
    ],
)
def test_cli_compose_output_is_not_an_input(tmp_path, output, what):
    recipe = toy_in(tmp_path)
    (tmp_path / "sub").mkdir()
    before = contents(tmp_path)
    out = tmp_path / output
    result = runner.invoke(app, ["compose", str(recipe), "-o", str(out)])
    assert result.exit_code != 0
    assert strip_ansi(result.output).strip() == (
        f"Not writing {out}: it is {what}. Choose another output with -o."
    )
    assert contents(tmp_path) == before
