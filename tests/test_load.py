from pathlib import Path

import pytest
from gotranx.load import load_ode, ode_from_string
from gotranx.exceptions import ODEFileNotFound, InvalidODEException, ComponentNotCompleteError


@pytest.fixture
def path():
    _path = Path("tmp_ode_file.ode")
    yield _path
    _path.unlink()


def test_load_nonexisting_ode_raises_ODEFileNotFound():
    with pytest.raises(ODEFileNotFound):
        load_ode("invalid_ode.ode")


def test_load_ode(path):
    path.write_text(
        """
    states("First component", "X-gate", x = 1, xr=3.14)
    states("First component", "Y-gate", y = 1)
    states("Second component", z=1)
    parameters("First component", a=1, b=2)
    parameters("Second component", c=3)

    expressions("First component")
    d = a + b * 2 - 3 / c

    expressions("First component", "X-gate")
    dx_dt=a+1
    dxr_dt = (x / b) - d * xr

    expressions("First component", "Y-gate")
    dy_dt = 2 * d - 1

    expressions("Second component")
    dz_dt = 1 + x - y
    """,
    )

    ode = load_ode(path)
    assert ode.num_components == 4
    assert set([c.name for c in ode.components]) == {
        "First component",
        "Second component",
        "X-gate",
        "Y-gate",
    }
    assert ode.num_states == 4
    assert ode.num_parameters == 3


def test_load_invalid_ode():
    with pytest.raises(InvalidODEException):
        ode_from_string("states(x=1)")


def test_load_incomplete_ode():
    with pytest.raises(ComponentNotCompleteError):
        ode_from_string("states(x=1, y=2)\ndx_dt = x + y")
