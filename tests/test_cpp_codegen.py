import pytest
import sympy
from sympy.codegen.ast import Assignment

from gotranx.cli import gotran2cpp
from gotranx.codegen import CFormat
from gotranx.schemes import Scheme
from gotranx.codegen.cpp import CppCodeGenerator, GotranCppCodePrinter
from gotranx.exceptions import InvalidNamespace
from gotranx.ode import make_ode

x, y = sympy.symbols("x y")


def test_cpp_printer_uses_std_math_and_fmax():
    p = GotranCppCodePrinter()
    assert p.doprint(sympy.exp(x)) == "std::exp(x)"
    # std::max(0, V) does not compile (int vs double), so Max stays fmax
    assert p.doprint(sympy.Max(0, x)) == "std::fmax(0, x)"
    assert p.doprint(sympy.Min(y, 1)) == "std::fmin(1, y)"


def test_cpp_printer_keeps_the_piecewise_assignment_ternary():
    expr = sympy.Piecewise((x, x > 0), (2 * x, True))
    assert GotranCppCodePrinter().doprint(Assignment(y, expr)) == "y = (x > 0) ? x : 2*x;"


def test_cpp_printer_double_literals():
    p = GotranCppCodePrinter()
    assert p.doprint(sympy.Float(2.5) * x) == "2.5*x"
    assert p.doprint(sympy.Rational(1, 3)) == "1.0/3.0"


@pytest.mark.parametrize(
    "expr, expected",
    [
        (sympy.Float(2.5) * x, "2.5f*x"),
        (x**2, "std::pow(x, 2.0f)"),
        (sympy.Rational(1, 3), "1.0f/3.0f"),
        (sympy.Max(0, x), "std::fmax(0.0f, x)"),
        (sympy.IndexedBase("states", shape=(3,))[0], "states[0]"),
    ],
)
def test_cpp_printer_float_literals(expr, expected):
    assert GotranCppCodePrinter(float_type="float").doprint(expr) == expected


def test_cpp_printer_renames_cxx_keywords():
    assert GotranCppCodePrinter().doprint(sympy.Symbol("new") * 2) == "2*new_"


@pytest.fixture(scope="module")
def ode(trans, parser):
    expr = """
    parameters(a=0)
    parameters("My component",
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0,
    beta=2.4
    )
    states("My component", x=1.0, y=2.0,z=3.05)

    expressions("My component")
    dy_dt = x*(rho - z) - y # millivolt
    dx_dt = sigma*(-x + y)
    dz_dt = -beta*z + x*y
    """
    tree = parser.parse(expr)
    return make_ode(*trans.transform(tree), name="lorentz")


@pytest.fixture(scope="module")
def codegen(ode) -> CppCodeGenerator:
    return CppCodeGenerator(ode, format=CFormat.none)


def test_cpp_rhs_signature(codegen):
    assert (
        "inline void rhs(const double t, const double *__restrict states, "
        "const double *__restrict parameters, double* values){"
    ) in codegen.rhs()


def test_cpp_initial_state_values(codegen):
    code = codegen.initial_state_values()
    assert "inline void init_state_values(double* states){" in code
    assert "states[0] = 1.0;" in code


def test_cpp_index_functions_are_inline_and_use_std_strcmp(codegen):
    code = codegen.state_index()
    assert "inline int state_index(const char name[])" in code
    assert 'std::strcmp(name, "x") == 0' in code


def test_cpp_constants(codegen):
    assert codegen.constants().splitlines() == [
        "constexpr int NUM_STATES = 3;",
        "constexpr int NUM_PARAMS = 4;",
        "constexpr int NUM_MONITORED = 3;",
    ]


def test_cpp_float_type(ode):
    codegen = CppCodeGenerator(ode, format=CFormat.none, float_type="float")
    code = codegen.rhs() + codegen.initial_state_values()
    assert "double" not in code
    assert "const float x = states[0];" in code
    assert "states[2] = 3.05f;" in code


def test_cpp_namespace_defaults_to_the_ode_name(ode):
    assert CppCodeGenerator(ode).namespace == "lorentz"
    assert CppCodeGenerator(ode, namespace="other").namespace == "other"


@pytest.mark.parametrize("name", ["1962_noble", "my.model", "my model", "default", "namespace"])
def test_cpp_invalid_namespace_is_rejected(ode, name):
    with pytest.raises(InvalidNamespace, match="--namespace"):
        CppCodeGenerator(ode, namespace=name)


def test_cpp_get_code_layout(ode):
    code = gotran2cpp.get_code(ode, scheme=list(Scheme), format=CFormat.none)
    assert code.splitlines()[0] == "#pragma once"
    assert "namespace lorentz {" in code
    assert "constexpr int NUM_STATES = 3;" in code
    assert "inline void forward_generalized_rush_larsen(" in code
    assert code.rstrip().endswith("}  // namespace lorentz")
    inside = code.split("namespace lorentz {", 1)[1]
    assert "#include" not in inside
