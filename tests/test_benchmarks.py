from pathlib import Path
import gotranx
import pytest

try:
    import myokit
except ImportError:
    myokit = None

here = Path(__file__).parent

cellml_file = here / "cellml_files" / "ToRORd_dynCl_mid.cellml"
ode_file = here / "odefiles" / "ToRORd_dyn_chloride.ode"


@pytest.fixture(scope="module", autouse=True)
def ode():
    yield gotranx.load_ode(ode_file)


@pytest.mark.skipif(myokit is None, reason="myokit not installed")
@pytest.mark.benchmark
def test_cell2gotran():
    gotranx.myokit.cellml_to_gotran(cellml_file)


@pytest.mark.benchmark
def test_save(ode, tmp_path):
    ode.save(tmp_path / "test.ode")


@pytest.mark.benchmark
def test_python_no_schemes(ode):
    gotranx.cli.gotran2py.get_code(ode)


@pytest.mark.benchmark
def test_python_explicit_euler(ode):
    gotranx.cli.gotran2py.get_code(ode, scheme=[gotranx.schemes.Scheme.explicit_euler])


@pytest.mark.benchmark
def test_python_generalized_rush_larsen(ode):
    gotranx.cli.gotran2py.get_code(ode, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen])


@pytest.mark.benchmark
def test_c_no_schemes(ode):
    gotranx.cli.gotran2c.get_code(ode)


@pytest.mark.benchmark
def test_c_explicit_euler(ode):
    gotranx.cli.gotran2c.get_code(ode, scheme=[gotranx.schemes.Scheme.explicit_euler])


@pytest.mark.benchmark
def test_c_generalized_rush_larsen(ode):
    gotranx.cli.gotran2c.get_code(ode, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen])


def _many_divisions(n):
    """n states, each rate divided by a sum of states: denominators the scan rejects."""
    states = ", ".join(f"y{i} = 0.5" for i in range(n))
    lines = ['parameters("m", k = 0.01)', f'states("m", {states})', 'expressions("m")']
    for i in range(n):
        j = (i + 1) % n
        lines.append(f"r{i} = y{i}/(y{i} + y{j} + 1.0)")
        lines.append(f"dy{i}_dt = -k*r{i} + k*r{j}")
    return "\n".join(lines) + "\n"


@pytest.mark.benchmark
def test_load_many_divisions():
    ode = gotranx.load.ode_from_string(_many_divisions(500))
    assert ode.num_states == 500
