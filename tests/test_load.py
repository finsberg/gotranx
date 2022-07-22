from pathlib import Path

import pytest
from gotran_parser import load_ode


@pytest.fixture
def path():
    _path = Path("tmp_ode_file.ode")
    yield _path
    _path.unlink()


def test_load_ode(path):
    path.write_text(
        """
    states("First component", "X-gate", x = 1, xr=3.14)
    states("First component", "Y-gate", y = 1)
    states("Second component", z=1, zi=0.0)
    parameters("First component", a=1, b=2)
    parameters("Second component", c=3)

    expressions("First component")
    dx_dt=a+1
    d = a + b * 2 - 3 / c
    dxr_dt= -x + xr
    dy_dt = 2 * d - 1

    expressions("Second component")
    dz_dt = 1 + x - y
    dzi_dt = z + zi
    """,
    )

    ode = load_ode(path)
    assert ode.num_components == 2
    assert ode.components[0].name == "First component"
    assert ode.components[1].name == "Second component"
