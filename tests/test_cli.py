import gotranx
import pytest
from pathlib import Path
from typer.testing import CliRunner

here = Path(__file__).parent.absolute()
runner = CliRunner(mix_stderr=False)


@pytest.fixture(scope="module")
def all_schemes():
    args = []
    for scheme in gotranx.schemes.Scheme:
        args.append("--scheme")
        args.append(scheme.value)
    return args


@pytest.fixture(scope="module")
def odefile(tmp_path_factory):
    expr = """
    parameters(a=0)
    parameters("My component",
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0,
    beta=2.4
    )
    states("My component", x=1.0, y=2.0,z=3.05)

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


def test_gotran2py_old(odefile, all_schemes):
    outfile = odefile.with_suffix(".py")

    result = runner.invoke(
        gotranx.cli.app, ["convert", str(odefile), "--to", ".py", "-o", str(outfile)] + all_schemes
    )
    assert result.exit_code == 0
    assert "lorentz.py" in result.stdout
    assert "lorentz" in result.stdout
    assert outfile.is_file()

    code = outfile.read_text()
    for scheme in gotranx.schemes.Scheme:
        assert scheme.value in code
    assert "rhs" in code
    assert "init_state_values" in code
    assert "init_parameter_values" in code
    assert "monitor_index" in code
    assert "state_index" in code
    assert "parameter_index" in code

    outfile.unlink()


def test_gotran2c_old(odefile, all_schemes):
    outfile = odefile.with_suffix(".h")
    result = runner.invoke(
        gotranx.cli.app, ["convert", str(odefile), "--to", ".h", "-o", str(outfile)] + all_schemes
    )
    assert result.exit_code == 0

    assert "lorentz.h" in result.stdout
    assert "lorentz" in result.stdout
    assert outfile.is_file()

    code = outfile.read_text()
    for scheme in gotranx.schemes.Scheme:
        assert scheme.value in code
    assert "rhs" in code
    assert "init_state_values" in code
    assert "init_parameter_values" in code
    assert "monitor_index" in code
    assert "state_index" in code
    assert "parameter_index" in code

    outfile.with_suffix(".h").unlink()


def test_cellml2ode_old():
    cellmlfile = here / "cellml_files" / "noble_1962.cellml"
    out_odefile = cellmlfile.with_suffix(".ode")
    result = runner.invoke(
        gotranx.cli.app,
        ["convert", str(cellmlfile), "-o", cellmlfile.with_suffix(".ode")],
    )
    assert result.exit_code == 0

    assert f"Wrote {out_odefile}" in result.stdout
    assert out_odefile.is_file()
    # Check that we can load the file
    ode = gotranx.load_ode(out_odefile)
    assert ode is not None
    out_odefile.unlink()


def test_cellml2ode():
    cellmlfile = here / "cellml_files" / "noble_1962.cellml"
    out_odefile = cellmlfile.with_suffix(".ode")
    result = runner.invoke(
        gotranx.cli.app,
        ["cellml2ode", str(cellmlfile), "-o", cellmlfile.with_suffix(".ode")],
    )
    assert result.exit_code == 0

    assert f"Wrote {out_odefile}" in result.stdout
    assert out_odefile.is_file()
    # Check that we can load the file
    ode = gotranx.load_ode(out_odefile)
    assert ode is not None
    out_odefile.unlink()
