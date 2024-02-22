from pathlib import Path

import pytest
from gotranx import load_ode
from gotranx import save
from gotranx.ode import make_ode


@pytest.fixture
def path():
    path_ = Path("tmp_file.ode")
    yield path_
    path_.unlink()


def test_write_ode_file_write_dummy(path, parser, trans):
    expr = """
    # This is an ODE.
    # Here is another line.
    # And here is a third line, just to make sure
    # that the total line length is more than 80 characters.
    states("First component", "X-gate", x = 1, xr=3.14)
    states("First component", "Y-gate", y = 1)
    states("Second component", z=1)
    parameters("First component", a=1, b=2)
    parameters("Second component", c=3)

    expressions("First component")
    d = a + b * 2 - 3 / c  # This is a comment

    expressions("First component", "X-gate")
    dx_dt=a+1  # This is another comment
    dxr_dt = (-d) * xr + (x / b) # mV

    expressions("First component", "Y-gate")
    dy_dt = 2 * d - 1 # ms

    expressions("Second component")
    dz_dt = 1 + x - y  # mM
    """
    tree = parser.parse(expr)
    old_ode = make_ode(*trans.transform(tree), name=path.stem)
    save.write_ODE_to_ode_file(old_ode, path)
    ode = load_ode(path)
    assert ode.simplify() == old_ode.simplify()


def test_write_ode_file_write_lorentz(path, parser, trans):
    expr = """
    parameters(
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0,
    beta=2.4
    )
    states(x=1.0, y=2.0,z=3.05)

    dx_dt = sigma*(-x + y)
    dy_dt = x*(rho - z) - y # millivolt
    dz_dt = -beta*z + x*y
    """
    tree = parser.parse(expr)
    old_ode = make_ode(*trans.transform(tree), name=path.stem)
    save.write_ODE_to_ode_file(old_ode, path)
    ode = load_ode(path)
    assert ode == old_ode
