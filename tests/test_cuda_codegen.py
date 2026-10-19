import pytest
import sympy

from gotranx.codegen.cuda import CUDACodePrinter

x = sympy.Symbol("x")


@pytest.mark.parametrize(
    "float_type, expr, expected",
    [
        ("double", sympy.IndexedBase("states", shape=(3,))[2], "states[2 * n_cells + cell]"),
        ("double", sympy.IndexedBase("values", shape=(3,))[0], "values[0 * n_cells + cell]"),
        (
            "double",
            sympy.IndexedBase("missing_variables", shape=(2,))[1],
            "missing_variables[1 * n_cells + cell]",
        ),
        ("double", sympy.IndexedBase("parameters", shape=(3,))[2], "parameters[2]"),
        ("float", sympy.IndexedBase("states", shape=(3,))[2], "states[2 * n_cells + cell]"),
        ("double", sympy.exp(x), "exp(x)"),
        # sqrt(5) would resolve to the constexpr host-only integer overload
        ("double", sympy.sqrt(5) * x, "sqrt(5.0)*x"),
        ("double", x**2, "pow(x, 2.0)"),
        ("double", sympy.Max(0, x), "fmax(0.0, x)"),
        ("float", sympy.sqrt(5) * x, "sqrt(5.0f)*x"),
        ("float", sympy.pi * x, "3.141592653589793f*x"),
    ],
)
def test_cuda_printer(float_type, expr, expected):
    assert CUDACodePrinter(float_type=float_type).doprint(expr) == expected
