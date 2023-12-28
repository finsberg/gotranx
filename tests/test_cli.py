import gotranx
import pytest
from typer.testing import CliRunner

runner = CliRunner(mix_stderr=False)


@pytest.fixture(scope="module")
def odefile(tmp_path_factory):
    expr = """
    parameters(a=0)
    parameters("My component",
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0,
    beta=2.4
    )
    states("My component", "info about states", x=1.0, y=2.0,z=3.05)

    expressions("My component")
    dy_dt = x*(rho - z) - y # millivolt
    dx_dt = sigma*(-x + y)
    dz_dt = -beta*z + x*y
    """
    fname = tmp_path_factory.mktemp("data") / "lorentz.ode"
    fname.write_text(expr)
    yield fname
    fname.unlink()


def test_cli_version():
    result = runner.invoke(gotranx.cli.app, "--version")
    assert result.exit_code == 0
    assert "gotranx" in result.stdout


def test_cli_license():
    result = runner.invoke(gotranx.cli.app, "--license")
    assert result.exit_code == 0
    assert "MIT" in result.stdout


def test_gotran2py(odefile):
    result = runner.invoke(gotranx.cli.app, [str(odefile), "--to", ".py"])
    assert result.exit_code == 0
    assert "lorentz.py" in result.stdout
    assert "lorentz" in result.stdout


def test_gotran2c(odefile):
    result = runner.invoke(gotranx.cli.app, [str(odefile), "--to", ".h"])
    assert result.exit_code == 0
    assert "lorentz.h" in result.stdout
    assert "lorentz" in result.stdout
