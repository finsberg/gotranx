"""Every model in `tests/odefiles`, generated as C and as C++, compiled and run.

The codegen tests pin what the backends print; these compile it, with warnings as errors,
and evaluate each model at its initial state against the Python backend: the initial
values, the index lookups, the right-hand side, the monitored values and one step of two
schemes. The C++ header is loaded through a small `extern "C"` shim, so both languages run
the same ctypes tests. They need a C and a C++ compiler (`conftest.c_compiler`,
`conftest.cxx_compiler`).
"""

import ctypes
from pathlib import Path

import numpy
import pytest

from gotranx.cli import gotran2c, gotran2cpp, gotran2py
from gotranx.codegen import CFormat
from gotranx.load import load_ode, ode_from_string
from gotranx.schemes import Scheme

HERE = Path(__file__).parent
# ohara_rudy_cipa_v1_2017.ode is a git-ignored local file that does not load
# (DuplicateSymbolError), so it only exists in some checkouts. Skip it until it is fixed.
ODEFILES = sorted(
    path for path in (HERE / "odefiles").glob("*.ode") if path.name != "ohara_rudy_cipa_v1_2017.ode"
)
CARDIAC_MODEL = HERE / "odefiles" / "ToRORd_dyn_chloride.ode"
SCHEMES = [Scheme.generalized_rush_larsen, Scheme.forward_explicit_euler]
LANGUAGES = ["c", "cpp"]
DT = 0.01
T = 0.0


def assert_close(actual, desired):
    # The same tolerance as tests/test_julia_runs.py: libm is not numpy, and the right-hand
    # side of a cell model at rest is a sum of currents that nearly cancel.
    numpy.testing.assert_allclose(actual, desired, rtol=1e-12, atol=1e-12)


def ptr(array: numpy.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def get_code(language: str, ode, **kwargs) -> str:
    backend = gotran2c if language == "c" else gotran2cpp
    return backend.get_code(ode, format=CFormat.none, **kwargs)


def shim(namespace: str, schemes: list[str]) -> str:
    """`extern "C"` wrappers with the C signatures, forwarding to the C++ namespace."""
    ns = namespace
    lines = [
        '#include "model.hpp"',
        'extern "C" {',
        f"void init_state_values(double* s) {{ {ns}::init_state_values(s); }}",
        f"void init_parameter_values(double* p) {{ {ns}::init_parameter_values(p); }}",
    ]
    lines += [
        f"int {kind}_index(const char* name) {{ return {ns}::{kind}_index(name); }}"
        for kind in ("state", "parameter", "monitor")
    ]
    lines += [
        f"void {name}(double t, const double* s, const double* p, double* v) "
        f"{{ {ns}::{name}(t, s, p, v); }}"
        for name in ("rhs", "monitor_values")
    ]
    lines += [
        f"void {name}(const double* s, double t, double dt, const double* p, double* v) "
        f"{{ {ns}::{name}(s, t, dt, p, v); }}"
        for name in schemes
    ]
    lines.append("}")
    return "\n".join(lines) + "\n"


def compile_code(code: str, language: str, compiler, directory: Path, schemes=None) -> Path:
    """Compile generated code to a shared library.

    For C++ with `schemes`, the library exports the C signatures (double precision) of the
    always-present functions and of `schemes` through a shim. Without, the source only
    includes the header, which still compiles every inline function in it."""
    if language == "c":
        source = directory / "model.c"
        source.write_text(code)
    else:
        (directory / "model.hpp").write_text(code)
        source = directory / "model.cpp"
        if schemes is None:
            source.write_text('#include "model.hpp"\n')
        else:
            namespace = code.split("namespace ", 1)[1].split()[0]
            source.write_text(shim(namespace, list(schemes)))
    return compiler.shared_library(source)


def load(library: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(library))
    for name in ("state_index", "parameter_index", "monitor_index"):
        getattr(lib, name).argtypes = [ctypes.c_char_p]
        getattr(lib, name).restype = ctypes.c_int
    return lib


@pytest.fixture(scope="module", params=LANGUAGES)
def toolchain(request):
    """The language, and a compiler for it. A missing C++ compiler only skips C++."""
    fixture = "c_compiler" if request.param == "c" else "cxx_compiler"
    return request.param, request.getfixturevalue(fixture)


@pytest.fixture(scope="module", params=ODEFILES, ids=lambda path: path.stem)
def ode(request):
    return load_ode(request.param)


@pytest.fixture(scope="module")
def library(ode, toolchain, tmp_path_factory) -> ctypes.CDLL:
    language, compiler = toolchain
    code = get_code(language, ode, scheme=list(Scheme))
    schemes = [s.value for s in Scheme]
    return load(compile_code(code, language, compiler, tmp_path_factory.mktemp(language), schemes))


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


def test_compiles_with_all_schemes(library):
    # The fixture compiled every scheme with warnings as errors
    assert library is not None


def test_initial_values_agree_with_python(library, python):
    states = numpy.zeros_like(python["states"])
    library.init_state_values(ptr(states))
    assert_close(states, python["states"])

    parameters = numpy.zeros_like(python["parameters"])
    library.init_parameter_values(ptr(parameters))
    assert_close(parameters, python["parameters"])


def test_index_functions_agree_with_python(library, python, ode):
    for state in ode.states:
        assert library.state_index(state.name.encode()) == python["state_index"](state.name)
    for parameter in ode.parameters:
        name = parameter.name
        assert library.parameter_index(name.encode()) == python["parameter_index"](name)
    for monitor in ode.intermediates + ode.state_derivatives:
        name = monitor.name
        assert library.monitor_index(name.encode()) == python["monitor_index"](name)
    assert library.state_index(b"no_such_name") == -1


def test_rhs_and_monitors_agree_with_python(library, python):
    states, parameters = python["states"], python["parameters"]

    out = numpy.zeros_like(python["rhs"])
    library.rhs(ctypes.c_double(T), ptr(states), ptr(parameters), ptr(out))
    assert_close(out, python["rhs"])

    out = numpy.zeros_like(python["monitors"])
    library.monitor_values(ctypes.c_double(T), ptr(states), ptr(parameters), ptr(out))
    assert_close(out, python["monitors"])


@pytest.mark.parametrize("scheme", [s.value for s in SCHEMES])
def test_schemes_agree_with_python(library, python, scheme):
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


def test_compiles_with_remove_unused(tor_ord, toolchain, tmp_path):
    language, compiler = toolchain
    code = get_code(language, tor_ord, scheme=list(Scheme), remove_unused=True)
    compile_code(code, language, compiler, tmp_path)


def test_compiles_hybrid_rush_larsen_with_stiff_states(tor_ord, toolchain, tmp_path):
    language, compiler = toolchain
    code = get_code(
        language, tor_ord, scheme=[Scheme.hybrid_rush_larsen], stiff_states=["v", "cai"]
    )
    compile_code(code, language, compiler, tmp_path)


def test_compiles_with_missing_values(tor_ord, toolchain, tmp_path):
    language, compiler = toolchain
    code = get_code(language, tor_ord, missing_values={"INa": 0, "cai": 1})
    assert "void missing_values(" in code
    compile_code(code, language, compiler, tmp_path)


# --- C++ only -----------------------------------------------------------------------------------


def test_cpp_float_compiles_without_promotion(ode, cxx_compiler, tmp_path):
    # In C++, std::pow(x, 2) and std::fmax(0, x) are double for a float x: every literal must
    # be a float, or these warnings (errors here) fire
    code = gotran2cpp.get_code(ode, scheme=list(Scheme), format=CFormat.none, float_type="float")
    strict = cxx_compiler.with_flags("-Wdouble-promotion", "-Wfloat-conversion")
    compile_code(code, "cpp", strict, tmp_path)


def test_cpp_two_models_link_into_one_program(cxx_compiler, tmp_path):
    for name in ("lorentz", "beeler_reuter_1977"):
        code = gotran2cpp.get_code(load_ode(HERE / "odefiles" / f"{name}.ode"), format=CFormat.none)
        (tmp_path / f"{name}.hpp").write_text(code)
    a = tmp_path / "a.cpp"
    a.write_text(
        '#include "lorentz.hpp"\n#include "beeler_reuter_1977.hpp"\n'
        'extern "C" int a() { return lorentz::NUM_STATES + beeler_reuter_1977::NUM_STATES; }\n'
    )
    b = tmp_path / "b.cpp"
    b.write_text('#include "lorentz.hpp"\nextern "C" int b() { return lorentz::NUM_STATES; }\n')

    lib = ctypes.CDLL(str(cxx_compiler.shared_library(a, b)))
    assert lib.a() == 3 + 8
    assert lib.b() == 3


def test_cpp_compiles_with_cxx_keywords_as_names(cxx_compiler, tmp_path):
    ode = ode_from_string(
        """
        parameters("m", new=1.0, delete=2.0)
        states("m", this=0.5, x=1.0)
        expressions("m")
        class = new*this
        dthis_dt = -class
        dx_dt = delete*x
        """
    )
    code = gotran2cpp.get_code(ode, scheme=list(Scheme), format=CFormat.none)
    assert "class_" in code
    assert "this_" in code
    compile_code(code, "cpp", cxx_compiler, tmp_path)


def test_cpp_compiles_with_missing_variables(cxx_compiler, tmp_path):
    from test_missing_variables import TEXT

    full = ode_from_string(TEXT)
    ode = full - full.get_component("inputs")
    # Subtracting a component names the ODE "ode - inputs", which is no namespace
    code = gotran2cpp.get_code(ode, scheme=list(Scheme), format=CFormat.none, namespace="split")
    assert "missing_variables" in code
    compile_code(code, "cpp", cxx_compiler, tmp_path)
