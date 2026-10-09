"""`Min` and `Max` in the `.ode` language, through every backend."""

import math

import numpy
import pytest
import sympy

import gotranx
from gotranx.cli import gotran2c, gotran2julia, gotran2mtk, gotran2py, gotran2ufl
from gotranx.codegen.c import Format as CFormat
from gotranx.load import ode_from_string
from gotranx.schemes import Scheme

MODEL = """
parameters("m", k = 0.5)
states("m", y = 0.5)
expressions("m")
r = k*Max(y - 0.2, 0.0) + Min(y, 1.0, k)
dy_dt = -r
"""
# both sides of every kink, and exactly on 0.2 and 1.0
Y = numpy.array([-1.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.9, 1.0, 1.5, 2.0])


def _generated(module, ode, **kwargs):
    namespace: dict = {}
    exec(compile(module.get_code(ode, **kwargs), f"<{module.__name__}>", "exec"), namespace)
    return namespace


def _expected_rhs(y, k=0.5):
    return -(k * numpy.maximum(y - 0.2, 0.0) + numpy.minimum(numpy.minimum(y, 1.0), k))


def test_min_max_parse_to_sympy():
    r = ode_from_string(MODEL)["r"].expr
    assert r.has(sympy.Max) and r.has(sympy.Min)


def test_names_starting_with_min_or_max_are_still_variables():
    ode = ode_from_string(
        "parameters(Max_rate = 1.0, Minimum = 2.0)\nstates(y = 0.0)\ndy_dt = Max_rate - Minimum*y\n"
    )
    assert {p.name for p in ode.parameters} == {"Max_rate", "Minimum"}


def test_numpy_rhs_is_numpy_minimum_maximum():
    ns = _generated(gotran2py, ode_from_string(MODEL))
    values = ns["rhs"](0.0, Y[None, :], ns["init_parameter_values"]())
    numpy.testing.assert_allclose(values[0], _expected_rhs(Y), rtol=1e-13, atol=1e-15)


def test_grl_through_max_is_exact_on_each_side():
    ode = ode_from_string("parameters(k = 2.0)\nstates(y = 0.0)\ndy_dt = -k*Max(y - 0.2, 0.0)\n")
    ns = _generated(gotran2py, ode, scheme=[Scheme.generalized_rush_larsen])
    y0 = numpy.array([[0.0, 0.1, 0.2, 0.5, 1.0]])
    # numpy.where evaluates both branches: the Rush-Larsen quotient is 0/0 where the rate is 0
    with numpy.errstate(divide="ignore", invalid="ignore"):
        y1 = ns["generalized_rush_larsen"](y0, 0.0, 0.1, ns["init_parameter_values"]())
    expected = numpy.where(y0 <= 0.2, y0, 0.2 + (y0 - 0.2) * math.exp(-0.2))
    numpy.testing.assert_allclose(y1, expected, rtol=1e-12)


def test_jax_prints_numpy_maximum():
    code = gotran2py.get_code(ode_from_string(MODEL), backend=gotran2py.Backend.jax)
    assert "numpy.maximum(" in code and "numpy.minimum(" in code
    assert "numpy.max(" not in code and "numpy.min(" not in code


def test_max_keeps_a_removable_pole_guarded():
    ode = ode_from_string(
        "states(y = -50.0)\nr = (y + 40.0)/(exp((y + 40.0)/10.0) - 1.0)*Max(y, -60.0)\ndy_dt = -r\n"
    )
    assert isinstance(ode["r"].expr, sympy.Piecewise)
    ns = _generated(gotran2py, ode)
    # numpy.where evaluates both branches: the unguarded one is 0/0 at the pole
    with numpy.errstate(divide="ignore", invalid="ignore"):
        values = ns["rhs"](0.0, numpy.array([[-40.0, -30.0]]), ns["init_parameter_values"]())
    # x/(exp(x/10) - 1) -> 10 at x = 0, times Max(-40, -60) = -40
    numpy.testing.assert_allclose(values[0], [400.0, 30.0 * 10.0 / (math.e - 1.0)], rtol=1e-6)


def test_myokit_export_of_min_max():
    pytest.importorskip("myokit")
    from gotranx.myokit import gotran_to_myokit

    model = gotran_to_myokit(ode_from_string(MODEL))
    r = next(v for v in model.variables(deep=True) if v.name() == "r")
    assert r.eval() == pytest.approx(-_expected_rhs(numpy.array(0.5)))  # y = 0.5, k = 0.5


def test_c_prints_fmin_fmax():
    # format=none: clang-format may be absent
    code = gotran2c.get_code(ode_from_string(MODEL), format=CFormat.none)
    assert "fmax(" in code and "fmin(" in code


def test_ufl_min_max_evaluate_like_numpy():
    pytest.importorskip("ufl")
    ode = ode_from_string(MODEL)
    ufl_ns = _generated(gotran2ufl, ode)
    params = [float(p) for p in _generated(gotran2py, ode)["init_parameter_values"]()]
    for y in (0.1, 0.2, 0.5, 1.5):
        value = float(ufl_ns["rhs"](0.0, [y], params)[0])
        assert value == pytest.approx(_expected_rhs(numpy.array(y)), rel=1e-12)


@pytest.mark.parametrize("module", [gotran2julia, gotran2mtk])
def test_julia_and_mtk_print_without_sympy_names(module):
    code = module.get_code(ode_from_string(MODEL))
    assert "Max(" not in code and "Min(" not in code


def test_save_and_load_round_trip(tmp_path):
    ode = ode_from_string(MODEL)
    ode.save(tmp_path / "m.ode")
    again = gotranx.load_ode(tmp_path / "m.ode")
    ns, ns2 = _generated(gotran2py, ode), _generated(gotran2py, again)
    p = ns["init_parameter_values"]()
    numpy.testing.assert_array_equal(ns["rhs"](0.0, Y[None, :], p), ns2["rhs"](0.0, Y[None, :], p))
