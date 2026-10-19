import pytest
import sympy

from gotranx.cli import gotran2cuda
from gotranx.codegen import CFormat
from gotranx.schemes import Scheme
from gotranx.codegen.cuda import CUDACodeGenerator, CUDACodePrinter
from gotranx.exceptions import ReservedName
from gotranx.load import ode_from_string
from gotranx.ode import make_ode
from gotranx.schemes import get_scheme

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
def codegen(ode) -> CUDACodeGenerator:
    return CUDACodeGenerator(ode, format=CFormat.none)


def test_cuda_rhs_kernel(codegen):
    code = codegen.rhs()
    assert (
        "static __global__ void rhs(const double t, const double *states, "
        "const double *__restrict__ parameters, double* values, "
        "const std::int64_t n_cells){"
    ) in code
    assert (
        "const std::int64_t cell = static_cast<std::int64_t>(blockIdx.x) * blockDim.x "
        "+ threadIdx.x;"
    ) in code
    assert "if (cell >= n_cells) return;" in code
    assert "const double x = states[0 * n_cells + cell];" in code
    assert "values[2 * n_cells + cell] = dz_dt;" in code


def test_cuda_scheme_kernel_signature(codegen):
    assert (
        "__global__ void forward_explicit_euler(const double *states, const double t, "
        "const double dt, const double *__restrict__ parameters, double* values, "
        "const std::int64_t n_cells){"
    ) in codegen.scheme(get_scheme("forward_explicit_euler"))


def test_cuda_init_state_values_is_a_kernel(codegen):
    code = codegen.initial_state_values()
    assert "__global__ void init_state_values(double* states, const std::int64_t n_cells){" in code
    assert "if (cell >= n_cells) return;" in code
    assert "states[2 * n_cells + cell] = 3.05;" in code


def test_cuda_host_functions(codegen):
    code = codegen.initial_parameter_values()
    assert "inline void init_parameter_values(double* parameters){" in code
    assert "parameters[3] = 12.0;" in code
    assert "inline int state_index(const char name[])" in codegen.state_index()


def test_cuda_missing_variables_come_before_n_cells():
    from test_missing_variables import TEXT

    full = ode_from_string(TEXT)
    ode = full - full.get_component("inputs")
    code = CUDACodeGenerator(ode, format=CFormat.none, namespace="split").rhs()
    assert "const double *__restrict__ missing_variables, const std::int64_t n_cells){" in code
    assert "const double u_in = missing_variables[0 * n_cells + cell];" in code


@pytest.mark.parametrize("name", ["cell", "n_cells"])
def test_cuda_reserved_names_are_rejected(name):
    ode = ode_from_string(
        f'parameters("m", {name}=1.0)\nstates("m", x=1.0)\nexpressions("m")\ndx_dt = -{name}*x\n'
    )
    with pytest.raises(ReservedName, match=name):
        CUDACodeGenerator(ode)


def test_cuda_get_code_layout(ode):
    code = gotran2cuda.get_code(ode, scheme=list(Scheme), format=CFormat.none)
    assert code.splitlines()[0] == "#pragma once"
    # nvcc includes it implicitly, and without it the header compiles on the host too
    assert "cuda_runtime.h" not in code
    assert "#include <cstdint>" in code
    assert "namespace lorentz {" in code
    assert "__global__ void forward_generalized_rush_larsen(" in code
    assert code.rstrip().endswith("}  // namespace lorentz")
