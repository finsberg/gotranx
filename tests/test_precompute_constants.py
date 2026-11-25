"""`precompute_constants`: the parameter-only intermediates, evaluated once.

With the option, the C and C++ backends move every intermediate that depends only on
parameters into a generated `init_constants`, which stores them in a `constants` array, and
`rhs`, `monitor_values` and the schemes read them from it instead of recomputing them.
"""

import pytest

from gotranx.codegen import CFormat
from gotranx.codegen.c import CCodeGenerator
from gotranx.codegen.cpp import CppCodeGenerator
from gotranx.exceptions import ReservedName
from gotranx.load import ode_from_string
from gotranx.schemes import get_scheme

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
