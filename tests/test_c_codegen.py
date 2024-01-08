# flake8: noqa: E501
import sys
from unittest import mock

import pytest
from gotranx.codegen import CCodeGenerator
from gotranx.codegen import RHSArgument
from gotranx.ode import make_ode


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
def codegen(ode) -> CCodeGenerator:
    return CCodeGenerator(ode)


def test_c_codegen_initial_state_values(codegen: CCodeGenerator):
    assert codegen.initial_state_values() == (
        "\nvoid init_state_values(double *states)\n"
        "{\n"
        "    /*\n"
        "    x=1.0, y=2.0, z=3.05\n"
        "    */\n"
        "    states[0] = 1.0;\n"
        "    states[1] = 2.0;\n"
        "    states[2] = 3.05;\n"
        "}\n"
    )


def test_c_codegen_initial_parameter_values(codegen: CCodeGenerator):
    assert codegen.initial_parameter_values() == (
        "\nvoid init_parameter_values(double *parameters)\n"
        "{\n"
        "    /*\n"
        "    a=0, beta=2.4, rho=21.0, sigma=12.0\n"
        "    */\n"
        "    parameters[0] = 0;\n"
        "    parameters[1] = 2.4;\n"
        "    parameters[2] = 21.0;\n"
        "    parameters[3] = 12.0;\n"
        "}\n"
    )


def test_c_codegen_initial_parameter_values_no_clang_format(ode):
    with mock.patch.dict(sys.modules, {"clang_format_docs": None}):
        codegen = CCodeGenerator(ode)
        assert codegen.initial_parameter_values() == (
            "\nvoid init_parameter_values(double* parameters){\n"
            "    /*\n"
            "    a=0, beta=2.4, rho=21.0, sigma=12.0\n"
            "    */\n"
            "    parameters[0] = 0;\n"
            "    parameters[1] = 2.4;\n"
            "    parameters[2] = 21.0;\n"
            "    parameters[3] = 12.0;\n"
            "}\n"
        )


@pytest.mark.parametrize(
    "order, arguments",
    [
        (
            RHSArgument.stp,
            (
                "const double *__restrict states, const double t, "
                "const double *__restrict parameters, double *values"
            ),
        ),
        (
            RHSArgument.spt,
            (
                "const double *__restrict states, "
                "const double *__restrict parameters, const double t, double *values"
            ),
        ),
        (
            RHSArgument.tsp,
            (
                "const double t, const double *__restrict states, "
                "const double *__restrict parameters, double *values"
            ),
        ),
    ],
)
def test_c_codegen_rhs(order: str, arguments: str, codegen: CCodeGenerator):
    assert codegen.rhs(order=order) == (
        f"\nvoid rhs({arguments})\n"
        "{\n"
        "\n"
        "    // Assign states\n"
        "    const double x = states[0];\n"
        "    const double y = states[1];\n"
        "    const double z = states[2];\n"
        "\n"
        "    // Assign parameters\n"
        "    const double a = parameters[0];\n"
        "    const double beta = parameters[1];\n"
        "    const double rho = parameters[2];\n"
        "    const double sigma = parameters[3];\n"
        "\n"
        "    // Assign expressions\n"
        "    const double dx_dt = sigma * (-x + y);\n"
        "    values[0] = dx_dt;\n"
        "    const double dy_dt = x * (rho - z) - y;\n"
        "    values[1] = dy_dt;\n"
        "    const double dz_dt = (-beta) * z + x * y;\n"
        "    values[2] = dz_dt;\n"
        "}\n"
    )


def test_c_codegen_forward_euler(codegen: CCodeGenerator):
    assert codegen.scheme("forward_euler") == (
        "\nvoid forward_euler(const double *__restrict states, const double t, const double dt,"
        "\n                   const double *__restrict parameters, double *values)"
        "\n{"
        "\n"
        "\n    // Assign states"
        "\n    const double x = states[0];"
        "\n    const double y = states[1];"
        "\n    const double z = states[2];"
        "\n"
        "\n    // Assign parameters"
        "\n    const double a = parameters[0];"
        "\n    const double beta = parameters[1];"
        "\n    const double rho = parameters[2];"
        "\n    const double sigma = parameters[3];"
        "\n"
        "\n    // Assign expressions"
        "\n    const double dx_dt = sigma * (-x + y);"
        "\n    values[0] = dt * dx_dt + x;"
        "\n    const double dy_dt = x * (rho - z) - y;"
        "\n    values[1] = dt * dy_dt + y;"
        "\n    const double dz_dt = (-beta) * z + x * y;"
        "\n    values[2] = dt * dz_dt + z;"
        "\n}"
        "\n"
    )


def test_c_codegen_forward_generalized_rush_larsen(codegen: CCodeGenerator):
    assert codegen.scheme("forward_generalized_rush_larsen") == (
        "\nvoid forward_generalized_rush_larsen(const double *__restrict states, const double t, const double dt,"
        "\n                                     const double *__restrict parameters, double *values)"
        "\n{"
        "\n"
        "\n    // Assign states"
        "\n    const double x = states[0];"
        "\n    const double y = states[1];"
        "\n    const double z = states[2];"
        "\n"
        "\n    // Assign parameters"
        "\n    const double a = parameters[0];"
        "\n    const double beta = parameters[1];"
        "\n    const double rho = parameters[2];"
        "\n    const double sigma = parameters[3];"
        "\n"
        "\n    // Assign expressions"
        "\n    const double dx_dt = sigma * (-x + y);"
        "\n    const double dx_dt_linearized = -sigma;"
        "\n    values[0] = x + ((fabs(dx_dt_linearized) > 1e-08) ? (dx_dt * (exp(dt * dx_dt_linearized) - 1) / dx_dt_linearized)"
        "\n                                                      : (dt * dx_dt));"
        "\n    const double dy_dt = x * (rho - z) - y;"
        "\n    const double dy_dt_linearized = -1;"
        "\n    values[1] = y + ((fabs(dy_dt_linearized) > 1e-08) ? (dy_dt * (exp(dt * dy_dt_linearized) - 1) / dy_dt_linearized)"
        "\n                                                      : (dt * dy_dt));"
        "\n    const double dz_dt = (-beta) * z + x * y;"
        "\n    const double dz_dt_linearized = -beta;"
        "\n    values[2] = z + ((fabs(dz_dt_linearized) > 1e-08) ? (dz_dt * (exp(dt * dz_dt_linearized) - 1) / dz_dt_linearized)"
        "\n                                                      : (dt * dz_dt));"
        "\n}"
        "\n"
    )
