"""A model with missing variables -- a component subtracted away -- in every backend.

Subtracting a component turns the names it defined into *missing variables*: values the
generated code takes as an argument. The Python backend has always handled them; these tests
hold the C, Julia and CytoZoo backends to the same contract, checked against the Python output.
"""

import ctypes
import shutil
import subprocess
import sys

import numpy
import pytest

from gotranx.cli import gotran2c, gotran2cytozoo, gotran2julia, gotran2py
from gotranx.codegen.c import Format as CFormat
from gotranx.load import ode_from_string
from gotranx.schemes import Scheme

TEXT = """
parameters("inputs", u_in = 2.0, w_in = 3.0)
parameters("m", k = 0.5)
states("m", v = 1.0, y = 0.25)
expressions("m")
r = k*y - u_in
s = w_in*v
dy_dt = -r
dv_dt = -s + u_in
"""
SCHEME = [Scheme.generalized_rush_larsen]
STATES = numpy.array([1.0, 0.25])  # v, y (reordered through state_index below)
MISSING = numpy.array([2.5, -1.5])  # u_in, w_in: missing_variables is ordered by name


@pytest.fixture(scope="module")
def ode():
    full = ode_from_string(TEXT)
    return full - full.get_component("inputs")


@pytest.fixture(scope="module")
def python(ode):
    namespace: dict = {}
    exec(gotran2py.get_code(ode, scheme=SCHEME), namespace)
    return namespace


def _reference(python):
    states = numpy.zeros(2)
    states[python["state_index"]("v")] = STATES[0]
    states[python["state_index"]("y")] = STATES[1]
    parameters = python["init_parameter_values"]()
    rhs = python["rhs"](0.0, states, parameters, MISSING)
    monitors = python["monitor_values"](0.0, states, parameters, MISSING)
    step = python["generalized_rush_larsen"](states, 0.0, 0.1, parameters, MISSING)
    return states, parameters, rhs, monitors, step


def test_the_model_has_two_missing_variables(ode):
    assert ode.missing_variables == {"u_in": 0, "w_in": 1}


# --- C -----------------------------------------------------------------------------------------


def test_c_has_a_missing_index_and_one_monitor_index(ode):
    code = gotran2c.get_code(ode, scheme=SCHEME, format=CFormat.none)
    assert code.count("int monitor_index(const char name[])") == 1
    assert code.count("int missing_index(const char name[])") == 1


def test_c_takes_and_unpacks_missing_variables(ode):
    code = gotran2c.get_code(ode, scheme=SCHEME, format=CFormat.none)
    for function in ("void rhs(", "void monitor_values(", "void generalized_rush_larsen("):
        signature = next(line for line in code.splitlines() if line.startswith(function))
        assert "const double *__restrict missing_variables" in signature, signature
    assert "const double u_in = missing_variables[0];" in code
    assert "const double w_in = missing_variables[1];" in code


def _compiler():
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


@pytest.mark.skipif(_compiler() is None or sys.platform == "win32", reason="needs a C compiler")
def test_c_compiles_and_agrees_with_python(ode, python, tmp_path):
    source = tmp_path / "model.c"
    library = tmp_path / "model.so"
    source.write_text(gotran2c.get_code(ode, scheme=SCHEME, format=CFormat.none))
    subprocess.run(
        [_compiler(), "-shared", "-fPIC", "-o", str(library), str(source), "-lm"], check=True
    )
    lib = ctypes.CDLL(str(library))
    for name in ("state_index", "monitor_index", "missing_index"):
        getattr(lib, name).argtypes = [ctypes.c_char_p]
        getattr(lib, name).restype = ctypes.c_int

    assert lib.missing_index(b"u_in") == 0
    assert lib.missing_index(b"w_in") == 1
    assert lib.missing_index(b"r") == -1
    assert lib.monitor_index(b"r") == python["monitor_index"]("r")
    assert lib.monitor_index(b"u_in") == -1

    states, parameters, rhs, monitors, step = _reference(python)

    def ptr(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    out = numpy.zeros_like(rhs)
    lib.rhs(ctypes.c_double(0.0), ptr(states), ptr(parameters), ptr(out), ptr(MISSING))
    numpy.testing.assert_allclose(out, rhs, rtol=1e-14)

    out = numpy.zeros_like(monitors)
    lib.monitor_values(ctypes.c_double(0.0), ptr(states), ptr(parameters), ptr(out), ptr(MISSING))
    numpy.testing.assert_allclose(out, monitors, rtol=1e-14)

    out = numpy.zeros_like(step)
    lib.generalized_rush_larsen(
        ptr(states),
        ctypes.c_double(0.0),
        ctypes.c_double(0.1),
        ptr(parameters),
        ptr(out),
        ptr(MISSING),
    )
    numpy.testing.assert_allclose(out, step, rtol=1e-14)


# --- Julia -------------------------------------------------------------------------------------


def test_julia_has_a_missing_index_and_one_monitor_index(ode):
    code = gotran2julia.get_code(ode, scheme=SCHEME)
    assert code.count("function monitor_index(name::String)") == 1
    assert code.count("function missing_index(name::String)") == 1


def test_julia_unpacks_missing_variables_one_based(ode):
    code = gotran2julia.get_code(ode, scheme=SCHEME)
    assert "u_in = missing_variables[1]" in code
    assert "w_in = missing_variables[2]" in code
    assert "missing_variables[0]" not in code


@pytest.mark.skipif(shutil.which("julia") is None, reason="needs julia on PATH")
@pytest.mark.parametrize("type_stable", [False, True])
def test_julia_runs_and_agrees_with_python(ode, python, tmp_path, type_stable):
    states, parameters, rhs, monitors, step = _reference(python)
    model = tmp_path / "model.jl"
    model.write_text(gotran2julia.get_code(ode, scheme=SCHEME, type_stable=type_stable))

    def vec(a):
        return "[" + ", ".join(repr(float(x)) for x in a) + "]"

    driver = tmp_path / "driver.jl"
    driver.write_text(
        f"""
        include("{model.as_posix()}")
        s = {vec(states)}; p = {vec(parameters)}; m = {vec(MISSING)}
        out = zeros({len(rhs)}); rhs(0.0, s, p, out, m); println(join(out, " "))
        mon = zeros({len(monitors)}); monitor_values(0.0, s, p, mon, m); println(join(mon, " "))
        st = zeros({len(step)}); generalized_rush_larsen(s, 0.0, 0.1, p, st, m)
        println(join(st, " "))
        println(missing_index("u_in"), " ", missing_index("w_in"), " ", monitor_index("r"))
        """
    )
    result = subprocess.run(
        ["julia", "--startup-file=no", str(driver)], capture_output=True, text=True, check=True
    )
    lines = result.stdout.strip().splitlines()
    numpy.testing.assert_allclose([float(x) for x in lines[0].split()], rhs, rtol=1e-14)
    numpy.testing.assert_allclose([float(x) for x in lines[1].split()], monitors, rtol=1e-14)
    numpy.testing.assert_allclose([float(x) for x in lines[2].split()], step, rtol=1e-14)
    # Julia indices are one-based
    assert lines[3].split() == ["1", "2", str(python["monitor_index"]("r") + 1)]


# --- CytoZoo -----------------------------------------------------------------------------------


def test_cytozoo_appends_missing_variables_as_trailing_parameters(ode):
    code = gotran2cytozoo.get_code(ode, model_name="Toy")
    assert "const NUM_MISSING = 2;" in code
    # Parameters first, then the missing variables in missing_index order, so `connect` and
    # parameter_index reach them by name.
    assert "TOY_PARAMETER_NAMES = (:k, :u_in, :w_in,)" in code
    assert "num_parameters(::Toy) = NUM_PARAMS + NUM_MISSING" in code


def test_cytozoo_inputs_start_as_nan(ode):
    code = gotran2cytozoo.get_code(ode, model_name="Toy")
    # Once per constructor that builds the vector: an unconnected input must poison the RHS.
    assert code.count("p[(NUM_PARAMS + 1):end] .= NaN") == 2


def test_cytozoo_passes_the_trailing_slice_as_missing_variables(ode):
    code = gotran2cytozoo.get_code(ode, model_name="Toy")
    inputs = "view(model.parameters, (NUM_PARAMS + 1):(NUM_PARAMS + NUM_MISSING))"
    assert f"rhs(t, u, model.parameters, du, nothing, nothing, {inputs})" in code
    assert f"rhs(t, u, model.parameters, du, p.x, p.overrides, {inputs})" in code
    assert f"monitor_values(t, u, model.parameters, mon, nothing, nothing, {inputs})" in code


def test_cytozoo_converts_missing_variables_to_the_state_type(ode):
    code = gotran2cytozoo.get_code(ode, model_name="Toy")
    # Like the parameters, whose vector they share: a model built in Float64 and evaluated
    # on Float32 states must not compute the terms that read an input in Float64.
    assert code.count("u_in = TYPE(missing_variables[1])") == 2  # rhs and monitor_values
    assert code.count("w_in = TYPE(missing_variables[2])") == 2
    assert "= missing_variables[" not in code


def test_cytozoo_without_missing_variables_is_unchanged(parser, trans):
    from gotranx.ode import make_ode

    ode = make_ode(*trans.transform(parser.parse(TEXT)))
    code = gotran2cytozoo.get_code(ode, model_name="Toy")
    assert "NUM_MISSING" not in code
    assert "NaN" not in code
    assert "num_parameters(::Toy) = NUM_PARAMS\n" in code
