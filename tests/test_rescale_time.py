import math
from pathlib import Path

import pytest
import sympy as sp

import gotranx
from gotranx.load import ode_from_string

here = Path(__file__).parent
TOY = here / "odefiles" / "contraction_toy.ode"

TEXT = """
states("a", y=1.0)
states("b", z=1.0)
expressions("a")
x = sin(t)
r = 2*y
dy_dt = -y
expressions("b")
dz_dt = -z
"""


def test_derivatives_scale_by_k():
    ode = gotranx.load_ode(TOY, remove_singularities=False)
    ms = ode.rescale_time(0.001)
    for d in ode.state_derivatives:
        assert sp.simplify(ms[d.name].expr - 0.001 * d.expr) == 0


def test_t_becomes_k_t():
    ode = ode_from_string(TEXT, name="m")
    ms = ode.rescale_time(0.001)
    assert ms["x"].expr == sp.sin(0.001 * ms.t)


def test_only_named_components():
    ode = ode_from_string(TEXT, name="m")
    ms = ode.rescale_time(0.5, components=["a"])
    assert sp.simplify(ms["dy_dt"].expr - 0.5 * ode["dy_dt"].expr) == 0
    assert sp.simplify(ms["dz_dt"].expr - ode["dz_dt"].expr) == 0


def test_components_may_be_any_iterable():
    ode = ode_from_string(TEXT, name="m")
    ms = ode.rescale_time(0.5, components=(c for c in ["a"]))
    assert ms == ode.rescale_time(0.5, components=["a"])
    assert sp.simplify(ms["dy_dt"].expr - 0.5 * ode["dy_dt"].expr) == 0


def test_intermediates_that_are_rates_are_untouched():
    ode = ode_from_string(TEXT, name="m")
    ms = ode.rescale_time(0.001)
    assert sp.simplify(ms["r"].expr - 2 * ms.symbols["y"]) == 0


def test_original_not_mutated():
    ode = ode_from_string(TEXT, name="m")
    before = ode["dy_dt"].expr
    ode.rescale_time(0.001)
    assert ode["dy_dt"].expr == before


@pytest.mark.parametrize("k", [0, -1, float("inf"), float("nan")])
def test_bad_k_raises(k):
    ode = ode_from_string(TEXT, name="m")
    assert not (k > 0 and math.isfinite(k))
    with pytest.raises(ValueError, match="positive and finite"):
        ode.rescale_time(k)


def test_unknown_components_raise():
    # A typo used to return the model unscaled
    ode = ode_from_string(TEXT, name="m")
    with pytest.raises(ValueError, match="^m has no component A, c: its components are a and b$"):
        ode.rescale_time(0.5, components=["a", "A", "c"])


def test_components_must_not_be_a_string():
    ode = ode_from_string(TEXT, name="m")
    with pytest.raises(TypeError, match="components must be a sequence of names"):
        ode.rescale_time(0.5, components="a")
