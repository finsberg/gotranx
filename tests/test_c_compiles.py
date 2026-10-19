"""Every model in `tests/odefiles`, generated as C, compiled and run.

The C codegen tests pin what the backend prints; these compile it, with warnings as errors,
and evaluate each model at its initial state against the Python backend: the initial
values, the index lookups, the right-hand side, the monitored values and one step of two
schemes. They need a C compiler (`conftest.c_compiler`).
"""

import ctypes
from pathlib import Path

import numpy
import pytest

from gotranx.cli import gotran2c, gotran2py
from gotranx.codegen import CFormat
from gotranx.load import load_ode
from gotranx.schemes import Scheme

HERE = Path(__file__).parent
# ohara_rudy_cipa_v1_2017.ode is a git-ignored local file that does not load
# (DuplicateSymbolError), so it only exists in some checkouts. Skip it until it is fixed.
ODEFILES = sorted(
    path for path in (HERE / "odefiles").glob("*.ode") if path.name != "ohara_rudy_cipa_v1_2017.ode"
)
CARDIAC_MODEL = HERE / "odefiles" / "ToRORd_dyn_chloride.ode"
SCHEMES = [Scheme.generalized_rush_larsen, Scheme.forward_explicit_euler]
DT = 0.01
T = 0.0


def assert_close(actual, desired):
    # The same tolerance as tests/test_julia_runs.py: libm is not numpy, and the right-hand
    # side of a cell model at rest is a sum of currents that nearly cancel.
    numpy.testing.assert_allclose(actual, desired, rtol=1e-12, atol=1e-12)


def ptr(array: numpy.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def compile_code(code: str, c_compiler, directory: Path) -> ctypes.CDLL:
    source = directory / "model.c"
    source.write_text(code)
    library = ctypes.CDLL(str(c_compiler.shared_library(source)))
    for name in ("state_index", "parameter_index", "monitor_index"):
        getattr(library, name).argtypes = [ctypes.c_char_p]
        getattr(library, name).restype = ctypes.c_int
    return library


@pytest.fixture(scope="module", params=ODEFILES, ids=lambda path: path.stem)
def ode(request):
    return load_ode(request.param)


@pytest.fixture(scope="module")
def library(ode, c_compiler, tmp_path_factory) -> ctypes.CDLL:
    code = gotran2c.get_code(ode, scheme=list(Scheme), format=CFormat.none)
    return compile_code(code, c_compiler, tmp_path_factory.mktemp("c"))


@pytest.fixture(scope="module")
def python(ode) -> dict:
    """The Python backend's values at the model's initial state."""
    namespace: dict = {}
    exec(gotran2py.get_code(ode, scheme=SCHEMES), namespace)
    states = namespace["init_state_values"]()
    parameters = namespace["init_parameter_values"]()
    return {
        "states": states,
        "parameters": parameters,
        "rhs": namespace["rhs"](T, states, parameters),
        "monitors": namespace["monitor_values"](T, states, parameters),
        "generalized_rush_larsen": namespace["generalized_rush_larsen"](states, T, DT, parameters),
        "forward_explicit_euler": namespace["forward_explicit_euler"](states, T, DT, parameters),
        "state_index": namespace["state_index"],
        "parameter_index": namespace["parameter_index"],
        "monitor_index": namespace["monitor_index"],
    }


def test_c_compiles_with_all_schemes(library):
    # The fixture compiled every scheme with warnings as errors
    assert library is not None


def test_c_initial_values_agree_with_python(library, python):
    states = numpy.zeros_like(python["states"])
    library.init_state_values(ptr(states))
    assert_close(states, python["states"])

    parameters = numpy.zeros_like(python["parameters"])
    library.init_parameter_values(ptr(parameters))
    assert_close(parameters, python["parameters"])


def test_c_index_functions_agree_with_python(library, python, ode):
    for state in ode.states:
        assert library.state_index(state.name.encode()) == python["state_index"](state.name)
    for parameter in ode.parameters:
        name = parameter.name
        assert library.parameter_index(name.encode()) == python["parameter_index"](name)
    for monitor in ode.intermediates + ode.state_derivatives:
        name = monitor.name
        assert library.monitor_index(name.encode()) == python["monitor_index"](name)
    assert library.state_index(b"no_such_name") == -1


def test_c_rhs_and_monitors_agree_with_python(library, python):
    states, parameters = python["states"], python["parameters"]

    out = numpy.zeros_like(python["rhs"])
    library.rhs(ctypes.c_double(T), ptr(states), ptr(parameters), ptr(out))
    assert_close(out, python["rhs"])

    out = numpy.zeros_like(python["monitors"])
    library.monitor_values(ctypes.c_double(T), ptr(states), ptr(parameters), ptr(out))
    assert_close(out, python["monitors"])


@pytest.mark.parametrize("scheme", [s.value for s in SCHEMES])
def test_c_schemes_agree_with_python(library, python, scheme):
    states, parameters = python["states"], python["parameters"]
    out = numpy.zeros_like(states)
    getattr(library, scheme)(
        ptr(states), ctypes.c_double(T), ctypes.c_double(DT), ptr(parameters), ptr(out)
    )
    assert_close(out, python[scheme])


# --- Options that take other code paths, compiled for one cardiac model ----------------------


@pytest.fixture(scope="module")
def tor_ord():
    return load_ode(CARDIAC_MODEL)


def test_c_compiles_with_remove_unused(tor_ord, c_compiler, tmp_path):
    code = gotran2c.get_code(tor_ord, scheme=list(Scheme), remove_unused=True, format=CFormat.none)
    compile_code(code, c_compiler, tmp_path)


def test_c_compiles_hybrid_rush_larsen_with_stiff_states(tor_ord, c_compiler, tmp_path):
    code = gotran2c.get_code(
        tor_ord,
        scheme=[Scheme.hybrid_rush_larsen],
        stiff_states=["v", "cai"],
        format=CFormat.none,
    )
    compile_code(code, c_compiler, tmp_path)


def test_c_compiles_with_missing_values(tor_ord, c_compiler, tmp_path):
    code = gotran2c.get_code(tor_ord, missing_values={"INa": 0, "cai": 1}, format=CFormat.none)
    assert "void missing_values(" in code
    compile_code(code, c_compiler, tmp_path)
