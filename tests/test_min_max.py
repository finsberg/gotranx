"""`Min` and `Max` in the `.ode` language, through every backend."""

import math

import lark
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
r = k*Max(y - 0.2, 0.0, 2.0*y - 3.0) + Min(y, 1.0, k)
dy_dt = -r
"""
# Both sides of every kink, and exactly on each: Max switches at 0.2 and 2.8, and Min at y = 1.0
# and y = k. With k = 0.5 the 1.0 in Min is never the smallest, so the per-cell test uses k = 1.5
# as well.
Y = numpy.array([-1.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.9, 1.0, 1.5, 2.0, 2.8, 3.0, 3.5])


def _generated(module, ode, **kwargs):
    namespace: dict = {}
    exec(compile(module.get_code(ode, **kwargs), f"<{module.__name__}>", "exec"), namespace)
    return namespace


def _expected_rhs(y, k=0.5):
    largest = numpy.maximum(numpy.maximum(y - 0.2, 0.0), 2.0 * y - 3.0)
    smallest = numpy.minimum(numpy.minimum(y, 1.0), k)
    return -(k * largest + smallest)


def test_min_max_parse_to_sympy():
    r = ode_from_string(MODEL)["r"].expr
    assert r.has(sympy.Max) and r.has(sympy.Min)


def test_names_starting_with_min_or_max_are_still_variables():
    ode = ode_from_string(
        "parameters(Max_rate = 1.0, Minimum = 2.0, min = 3.0, max = 4.0)\n"
        "states(y = 0.0)\n"
        "dy_dt = Max_rate - Minimum*y + max - min\n"
    )
    assert {p.name for p in ode.parameters} == {"Max_rate", "Minimum", "min", "max"}


def test_min_and_max_are_reserved_in_expressions():
    with pytest.raises(lark.exceptions.UnexpectedToken):
        ode_from_string("parameters(Max = 1.0)\nstates(y = 0.0)\ndy_dt = Max*y\n")


def test_numpy_rhs_is_numpy_minimum_maximum():
    ns = _generated(gotran2py, ode_from_string(MODEL))
    values = ns["rhs"](0.0, Y[None, :], ns["init_parameter_values"]())
    numpy.testing.assert_allclose(values[0], _expected_rhs(Y), rtol=1e-13, atol=1e-15)


def test_numpy_rhs_with_per_cell_parameters():
    ns = _generated(gotran2py, ode_from_string(MODEL))
    y, k = (a.ravel() for a in numpy.meshgrid(Y, [0.5, 1.5]))  # k on both sides of 1.0
    parameters = numpy.repeat(ns["init_parameter_values"]()[:, None], y.size, axis=1)
    parameters[ns["parameter_index"]("k")] = k
    values = ns["rhs"](0.0, y[None, :], parameters)
    numpy.testing.assert_allclose(values[0], _expected_rhs(y, k), rtol=1e-13, atol=1e-15)


def test_grl_through_max_is_exact_on_each_side():
    ode = ode_from_string(
        "parameters(k = 2.0, c = 0.5)\nstates(y = 0.0)\ndy_dt = c - k*Max(y - 0.2, 0.0)\n"
    )
    ns = _generated(gotran2py, ode, scheme=[Scheme.generalized_rush_larsen])
    y0 = numpy.array([[0.0, 0.1, 0.2, 0.5, 1.0]])
    dt, k, c = 0.1, 2.0, 0.5
    # numpy.where evaluates both branches: the Rush-Larsen quotient is 0/0 where the rate is 0
    with numpy.errstate(divide="ignore", invalid="ignore"):
        y1 = ns["generalized_rush_larsen"](y0, 0.0, dt, ns["init_parameter_values"]())
    # Flat side: the linearization is 0, so the step is forward Euler. Steep side: exact for the
    # linear dy/dt = c - k*(y - 0.2), whose fixed point is 0.2 + c/k.
    fixed = 0.2 + c / k
    expected = numpy.where(y0 < 0.2, y0 + c * dt, fixed + (y0 - fixed) * math.exp(-k * dt))
    off_kink = y0 != 0.2
    numpy.testing.assert_allclose(y1[off_kink], expected[off_kink], rtol=1e-12)
    assert numpy.isfinite(y1).all()


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


EXPORTED = [
    "k*Max(y - 0.2, 0.0, 2.0*y - 3.0) + Min(y, 1.0, k)",  # MODEL's r
    "Min(Max(y, 0.0), 1.0)",  # the clamp in docs/grammar.md
    "Max(Min(y, 1.0), k)",
    "Conditional(Ge(Max(y, 0.0), k), y, k)",
    "Max(Conditional(Gt(y, 0.0), y, -y), k)",
    "Max(y, 0.1, k, 2.0*y - 1.0)",  # four arguments
    "Min(y, 1.0, k, y*y, -y)",  # five
    "Conditional(And(Gt(y, 0.0), Lt(y, 1.0), Gt(k, 0.0)), y, k)",  # a three-way And
]


@pytest.mark.parametrize("expr", EXPORTED)
def test_myokit_export_evaluates_like_numpy(expr):
    pytest.importorskip("myokit")
    from gotranx.myokit import gotran_to_myokit

    ode = ode_from_string(
        f'parameters("m", k = 0.5)\nstates("m", y = 0.5)\nexpressions("m")\nr = {expr}\n'
        "dy_dt = -r\n"
    )
    model = gotran_to_myokit(ode)
    r = next(v for v in model.variables(deep=True) if v.name() == "r")
    y = next(v for v in model.variables(deep=True) if v.name() == "y")
    ns = _generated(gotran2py, ode)
    parameters = ns["init_parameter_values"]()
    for value in (-1.0, 0.0, 0.3, 0.5, 0.8, 1.0, 2.0, 3.0):
        y.set_initial_value(value)
        expected = -ns["rhs"](0.0, numpy.array([value]), parameters)[0]
        assert r.eval() == pytest.approx(expected, rel=1e-12, abs=1e-15), value


def test_cellml_export_of_the_clamp(tmp_path):
    pytest.importorskip("myokit")
    from gotranx.myokit import gotran_to_cellml

    ode = ode_from_string("states(y = 0.5)\nr = Min(Max(y, 0.0), 1.0)\ndy_dt = -r\n")
    gotran_to_cellml(ode, filename=tmp_path / "clamp.cellml")
    assert (tmp_path / "clamp.cellml").read_text().count("<piecewise>") >= 2


def test_c_prints_fmin_fmax():
    # format=none: clang-format may be absent
    code = gotran2c.get_code(ode_from_string(MODEL), format=CFormat.none)
    assert "fmax(" in code and "fmin(" in code


def test_ufl_min_max_evaluate_like_numpy():
    pytest.importorskip("ufl")
    ode = ode_from_string(MODEL)
    ufl_ns = _generated(gotran2ufl, ode)
    py_ns = _generated(gotran2py, ode)
    for k in (0.5, 1.5):  # k on both sides of 1.0, so each argument of Min is the smallest
        params = [float(p) for p in py_ns["init_parameter_values"](k=k)]
        for y in (0.1, 0.2, 0.5, 1.2, 1.5, 3.0):
            value = float(ufl_ns["rhs"](0.0, [y], params)[0])
            assert value == pytest.approx(_expected_rhs(numpy.array(y), k), rel=1e-12), (k, y)


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
