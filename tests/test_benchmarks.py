from pathlib import Path
import gotranx
import pytest

here = Path(__file__).parent

cellml_file = here / "cellml_files" / "ToRORd_dynCl_mid.cellml"
ode_file = here / "cellml_files" / "ToRORd_dynCl_mid.ode"


@pytest.fixture(scope="module", autouse=True)
def ode():
    _ode = gotranx.myokit.cellml_to_gotran(cellml_file)
    _ode.save(ode_file)
    yield gotranx.load_ode(ode_file)
    ode_file.unlink()


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
