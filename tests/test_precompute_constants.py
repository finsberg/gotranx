"""`precompute_constants`: the parameter-only intermediates, evaluated once.

With the option, the C and C++ backends move every intermediate that depends only on
parameters into a generated `init_constants`, which stores them in a `constants` array, and
`rhs`, `monitor_values` and the schemes read them from it instead of recomputing them.
"""

import ctypes

import numpy
import pytest

from gotranx.cli import gotran2c, gotran2cpp
from gotranx.codegen import CFormat
from gotranx.codegen.c import CCodeGenerator
from gotranx.codegen.cpp import CppCodeGenerator
from gotranx.exceptions import ReservedName
from gotranx.load import load_ode, ode_from_string
from gotranx.schemes import Scheme, get_scheme
from test_compiles import CARDIAC_MODEL, DT, SCHEMES, T, compile_code, ptr

MODEL = """
parameters(a=2.0, b=3.0, flag=1.0)
states(x=1.0)
c = a*b
e = exp(c) + Conditional(Gt(flag, 0.5), a, b)
f = t*a
g = c + x
dx_dt = -e*x + f + g
"""


@pytest.fixture(scope="module")
def ode():
    return ode_from_string(MODEL, name="model")


def generator(ode, **kwargs) -> CCodeGenerator:
    return CCodeGenerator(ode, format=CFormat.none, precompute_constants=True, **kwargs)


def test_constant_index_is_parameter_only_intermediates(ode):
    assert generator(ode).constant_index == {"c": 0, "e": 1}


def test_constant_index_is_empty_when_off(ode):
    assert CCodeGenerator(ode, format=CFormat.none).constant_index == {}


def test_init_constants_is_empty_when_off(ode):
    assert CCodeGenerator(ode, format=CFormat.none).init_constants() == ""


def test_signatures_take_constants_after_parameters(ode):
    codegen = generator(ode)
    pair = "const double *__restrict parameters, const double *__restrict constants"
    assert pair in codegen.rhs()
    assert pair in codegen.monitor_values()
    assert pair in codegen.scheme(get_scheme("rush_larsen"))
    pts = codegen.rhs(order="pts")
    assert "void rhs(" + pair + ", const double t" in pts


def test_hoisted_names_are_read_from_constants(ode):
    rhs = generator(ode).rhs()
    assert "const double c = constants[0];" in rhs
    assert "const double e = constants[1];" in rhs
    assert "exp(" not in rhs


def test_scheme_reads_hoisted_names_from_constants(ode):
    code = generator(ode).scheme(get_scheme("rush_larsen"))
    assert "const double e = constants[1];" in code
    assert "exp(c)" not in code


def test_init_constants_evaluates_in_order(ode):
    code = generator(ode).init_constants()
    assert (
        "void init_constants(const double *__restrict parameters, double *__restrict constants)"
        in code
    )
    assert "exp(" in code
    assert code.index("constants[0] = c;") < code.index("constants[1] = e;")


def test_conditional_is_hoisted_as_ternary(ode):
    assert "?" in generator(ode).init_constants()


def test_reserved_name_constants():
    ode = ode_from_string(
        "parameters(constants=1.0)\nstates(x=1.0)\ndx_dt = -constants*x\n", name="model"
    )
    CCodeGenerator(ode, format=CFormat.none)  # fine without the option
    with pytest.raises(ReservedName):
        generator(ode)


def test_cpp_generator_passes_flag(ode):
    codegen = CppCodeGenerator(ode, format=CFormat.none, precompute_constants=True)
    assert "constexpr int NUM_CONSTANTS = 2;" in codegen.constants()
    assert "inline void init_constants(" in codegen.init_constants()


# --- Generated, compiled and run: the same numbers, bit for bit --------------------------------

#: Without contraction, the compiler cannot fuse a multiply in a hoisted expression with an
#: add after it, which it could do when both are in the same function
STRICT = ("-ffp-contract=off",)
STEPS = 1000


def get_code(language: str, ode, **kwargs) -> str:
    backend = gotran2c if language == "c" else gotran2cpp
    return backend.get_code(ode, format=CFormat.none, **kwargs)


def hoisted_shim(namespace: str, schemes: list[str]) -> str:
    """`extern "C"` wrappers for the C++ functions that take `constants`."""
    ns = namespace
    lines = [
        '#include "model.hpp"',
        'extern "C" {',
        f"void init_state_values(double* s) {{ {ns}::init_state_values(s); }}",
        f"void init_parameter_values(double* p) {{ {ns}::init_parameter_values(p); }}",
        f"void init_constants(const double* p, double* c) {{ {ns}::init_constants(p, c); }}",
    ]
    lines += [
        f"void {name}(double t, const double* s, const double* p, const double* c, double* v) "
        f"{{ {ns}::{name}(t, s, p, c, v); }}"
        for name in ("rhs", "monitor_values")
    ]
    lines += [
        f"void {name}(const double* s, double t, double dt, const double* p, const double* c, "
        f"double* v) {{ {ns}::{name}(s, t, dt, p, c, v); }}"
        for name in schemes
    ]
    lines.append("}")
    return "\n".join(lines) + "\n"


def build(language, compiler, directory, code, schemes, hoisted):
    directory.mkdir()
    if language == "cpp" and hoisted:
        (directory / "model.hpp").write_text(code)
        namespace = code.split("namespace ", 1)[1].split()[0]
        source = directory / "model.cpp"
        source.write_text(hoisted_shim(namespace, schemes))
        return ctypes.CDLL(str(compiler.shared_library(source)))
    return ctypes.CDLL(str(compile_code(code, language, compiler, directory, schemes)))


class Pair:
    """The plain and the hoisted build of one model, called with the same inputs."""

    def __init__(self, ode, language, compiler, directory, **kwargs):
        self.schemes = [s.value for s in kwargs.get("scheme", [])]
        plain = get_code(language, ode, **kwargs)
        hoisted = get_code(language, ode, precompute_constants=True, **kwargs)
        self.plain = build(language, compiler, directory / "plain", plain, self.schemes, False)
        self.hoisted = build(language, compiler, directory / "hoisted", hoisted, self.schemes, True)
        self.states = numpy.zeros(ode.num_states)
        self.parameters = numpy.zeros(ode.num_parameters)
        self.plain.init_state_values(ptr(self.states))
        self.plain.init_parameter_values(ptr(self.parameters))
        self.constants = numpy.zeros(max(1, len(ode.intermediates)))
        self.hoisted.init_constants(ptr(self.parameters), ptr(self.constants))

    def rhs(self, name, size, states):
        a, b = numpy.zeros(size), numpy.zeros(size)
        t = ctypes.c_double(T)
        p, c = ptr(self.parameters), ptr(self.constants)
        getattr(self.plain, name)(t, ptr(states), p, ptr(a))
        getattr(self.hoisted, name)(t, ptr(states), p, c, ptr(b))
        return a, b

    def step(self, name, states_a, states_b):
        a, b = numpy.zeros_like(states_a), numpy.zeros_like(states_b)
        t, dt = ctypes.c_double(T), ctypes.c_double(DT)
        p, c = ptr(self.parameters), ptr(self.constants)
        getattr(self.plain, name)(ptr(states_a), t, dt, p, ptr(a))
        getattr(self.hoisted, name)(ptr(states_b), t, dt, p, c, ptr(b))
        return a, b


@pytest.fixture(scope="module")
def tor_ord():
    return load_ode(CARDIAC_MODEL)


@pytest.fixture(scope="module", params=["c", "cpp"])
def strict_toolchain(request):
    fixture = "c_compiler" if request.param == "c" else "cxx_compiler"
    return request.param, request.getfixturevalue(fixture).with_flags(*STRICT)


@pytest.mark.parametrize("language", ["c", "cpp"])
def test_get_code_off_is_unchanged(tor_ord, language):
    scheme = [Scheme.rush_larsen]
    default = get_code(language, tor_ord, scheme=scheme)
    assert default == get_code(language, tor_ord, scheme=scheme, precompute_constants=False)
    assert "constants" not in default


@pytest.mark.parametrize("language", ["c", "cpp"])
def test_get_code_on_has_init_constants(tor_ord, language):
    code = get_code(language, tor_ord, scheme=[Scheme.rush_larsen], precompute_constants=True)
    assert "NUM_CONSTANTS = " in code
    assert "void init_constants(" in code


def assert_bit_identical(pair, ode):
    num_monitored = len(ode.intermediates) + len(ode.state_derivatives)
    numpy.testing.assert_array_equal(*pair.rhs("rhs", ode.num_states, pair.states))
    numpy.testing.assert_array_equal(*pair.rhs("monitor_values", num_monitored, pair.states))
    for name in pair.schemes:
        numpy.testing.assert_array_equal(*pair.step(name, pair.states, pair.states))
    a, b = pair.states.copy(), pair.states.copy()
    for _ in range(STEPS):
        a, b = pair.step("rush_larsen", a, b)
    numpy.testing.assert_array_equal(a, b)


def test_bit_identical_with_constants(tor_ord, strict_toolchain, tmp_path):
    language, compiler = strict_toolchain
    pair = Pair(tor_ord, language, compiler, tmp_path, scheme=SCHEMES)
    assert_bit_identical(pair, tor_ord)


def test_bit_identical_with_remove_unused(tor_ord, c_compiler, tmp_path):
    compiler = c_compiler.with_flags(*STRICT)
    pair = Pair(tor_ord, "c", compiler, tmp_path, scheme=[Scheme.rush_larsen], remove_unused=True)
    assert_bit_identical(pair, tor_ord)


@pytest.mark.parametrize("language", ["c", "cpp"])
def test_no_constants_compiles(language, request, tmp_path):
    ode = ode_from_string("parameters(a=1.0)\nstates(x=1.0)\ndx_dt = -a*x\n", name="model")
    code = get_code(language, ode, scheme=[Scheme.rush_larsen], precompute_constants=True)
    assert "NUM_CONSTANTS = 0" in code
    compiler = request.getfixturevalue("c_compiler" if language == "c" else "cxx_compiler")
    pair = Pair(ode, language, compiler, tmp_path, scheme=[Scheme.rush_larsen])
    plain, hoisted = numpy.zeros(1), numpy.zeros(1)
    pair.plain.rhs(ctypes.c_double(T), ptr(pair.states), ptr(pair.parameters), ptr(plain))
    pair.hoisted.rhs(ctypes.c_double(T), ptr(pair.states), ptr(pair.parameters), None, ptr(hoisted))
    numpy.testing.assert_array_equal(plain, hoisted)


def test_missing_values_with_constants(tor_ord, c_compiler, tmp_path):
    code = get_code("c", tor_ord, missing_values={"INa": 0, "cai": 1}, precompute_constants=True)
    assert "void missing_values(" in code
    assert "const double *__restrict constants" in code.split("void missing_values(", 1)[1]
    compile_code(code, "c", c_compiler, tmp_path)
