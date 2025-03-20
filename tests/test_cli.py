from textwrap import dedent
import gotranx
import pytest
from pathlib import Path
from typer.testing import CliRunner

try:
    import myokit
except ImportError:
    myokit = None

here = Path(__file__).parent.absolute()
runner = CliRunner(mix_stderr=False)


@pytest.fixture(scope="session")
def config_file(tmp_path_factory):
    text = dedent(
        """
        [tool.gotranx]
        scheme = ["hybrid_rush_larsen"]
        stiff_states = ["x", "y"]
        verbose = true
        delta = 1e-3

        [tool.gotranx.python]
        format = "ruff"

        [tool.gotranx.c]
        format = "none"
        to = ".c"
        """
    )
    fname = tmp_path_factory.mktemp("data") / "pyproject.toml"
    fname.write_text(text)
    yield fname
    fname.unlink()


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


@pytest.mark.skipif(myokit is None, reason="myokit not installed")
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


@pytest.mark.skipif(myokit is None, reason="myokit not installed")
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


@pytest.mark.parametrize("backend", gotranx.cli.gotran2py.Backend)
@pytest.mark.parametrize("format", gotranx.codegen.PythonFormat)
def test_gotran2py(backend, format, odefile, all_schemes):
    outfile = odefile.with_suffix(".py")

    stiff_states = ["-s", "x", "-s", "y", "-s", "w"]

    result = runner.invoke(
        gotranx.cli.app,
        ["ode2py", str(odefile), "-v", "-o", str(outfile), "-f", format.value, "-b", backend.value]
        + all_schemes
        + stiff_states,
    )
    assert result.exit_code == 0
    assert "lorentz.py" in result.stdout
    assert "lorentz" in result.stdout
    if format != gotranx.codegen.PythonFormat.none:
        assert "Applying formatter" in result.stdout
        assert format.value in result.stdout

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

    if backend == gotranx.cli.gotran2py.Backend.jax:
        assert "jax" in code

    outfile.unlink()


@pytest.mark.parametrize("format", gotranx.codegen.CFormat)
@pytest.mark.parametrize("suffix", [".h", ".c"])
def test_gotran2c(format, suffix, odefile, all_schemes):
    outfile = odefile.with_suffix(suffix)
    result = runner.invoke(
        gotranx.cli.app,
        ["ode2c", str(odefile), "-v", "--to", suffix, "-f", format.value, "-o", str(outfile)]
        + all_schemes,
    )
    assert result.exit_code == 0

    assert f"lorentz{suffix}" in result.stdout
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

    outfile.with_suffix(suffix).unlink()


def test_ode2py_config_file(odefile, config_file):
    outfile = odefile.with_suffix(".py")
    result = runner.invoke(
        gotranx.cli.app,
        ["ode2py", str(odefile), "-c", str(config_file), "-o", str(outfile)],
    )
    assert result.exit_code == 0
    assert "lorentz.py" in result.stdout
    assert "lorentz" in result.stdout
    assert "stiff_states=['x', 'y']" in result.stdout
    assert "Applying formatter" in result.stdout
    assert "ruff" in result.stdout
    assert outfile.is_file()
    code = outfile.read_text()
    assert "hybrid_rush_larsen" in code
    outfile.unlink()


def test_ode2c_config_file(odefile, config_file):
    outfile = odefile.with_suffix(".c")
    result = runner.invoke(
        gotranx.cli.app,
        ["ode2c", str(odefile), "-c", str(config_file), "-o", str(outfile)],
    )
    assert result.exit_code == 0
    assert "lorentz.c" in result.stdout
    assert "lorentz" in result.stdout
    assert "stiff_states=['x', 'y']" in result.stdout
    assert outfile.is_file()
    code = outfile.read_text()
    assert "hybrid_rush_larsen" in code
    outfile.unlink()


def test_ode2julia_config_file(odefile, config_file):
    outfile = odefile.with_suffix(".jl")
    result = runner.invoke(
        gotranx.cli.app,
        ["ode2julia", str(odefile), "-c", str(config_file), "-o", str(outfile)],
    )
    assert result.exit_code == 0
    assert "lorentz.jl" in result.stdout
    assert "lorentz" in result.stdout
    assert "stiff_states=['x', 'y']" in result.stdout
    assert outfile.is_file()
    code = outfile.read_text()
    assert "hybrid_rush_larsen" in code
    outfile.unlink()
