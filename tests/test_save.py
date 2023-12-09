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


def test_write_ode_file(path, parser, trans):
    expr = """
    parameters(a=0)
    parameters("My component",
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0,
    beta=2.4
    )
    states("My component", "info about states", x=1.0, y=2.0,z=3.05)

    expressions("My component")
    dx_dt = sigma*(-x + y)
    dy_dt = x*(rho - z) - y # millivolt
    dz_dt = -beta*z + x*y
    """
    tree = parser.parse(expr)
    old_ode = make_ode(*trans.transform(tree), name=path.stem)
    save.write_ODE_to_ode_file(old_ode, path)
    ode = load_ode(path)
    assert ode == old_ode
