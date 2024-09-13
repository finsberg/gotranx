# flake8: noqa: E501
import sys
from unittest import mock

import pytest
from gotranx.schemes import get_scheme
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


@pytest.mark.skipif(sys.platform == "win32", reason="clang-format-docs is not available on Windows")
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


@pytest.mark.skipif(sys.platform == "win32", reason="clang-format-docs is not available on Windows")
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


@pytest.mark.skipif(sys.platform == "win32", reason="clang-format-docs is not available on Windows")
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


@pytest.mark.skipif(sys.platform == "win32", reason="clang-format-docs is not available on Windows")
def test_c_codegen_forward_euler(codegen: CCodeGenerator):
    assert codegen.scheme(get_scheme("forward_euler")) == (
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


@pytest.mark.skipif(sys.platform == "win32", reason="clang-format-docs is not available on Windows")
def test_c_codegen_forward_generalized_rush_larsen(codegen: CCodeGenerator):
    assert codegen.scheme(get_scheme("forward_generalized_rush_larsen")) == (
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


@pytest.mark.skipif(sys.platform == "win32", reason="clang-format-docs is not available on Windows")
def test_c_remove_unused_rhs(ode_unused):
    codegen_orig = CCodeGenerator(ode_unused)
    assert codegen_orig.rhs() == (
        "\nvoid rhs(const double t, const double *__restrict states, const double *__restrict parameters, double *values)"
        "\n{"
        "\n"
        "\n    // Assign states"
        "\n    const double unused_state = states[0];"
        "\n    const double x = states[1];"
        "\n    const double y = states[2];"
        "\n    const double z = states[3];"
        "\n"
        "\n    // Assign parameters"
        "\n    const double a = parameters[0];"
        "\n    const double beta = parameters[1];"
        "\n    const double rho = parameters[2];"
        "\n    const double sigma = parameters[3];"
        "\n    const double unused_parameter = parameters[4];"
        "\n"
        "\n    // Assign expressions"
        "\n    const double unused_expression = 0;"
        "\n    const double dunused_state_dt = 0;"
        "\n    values[0] = dunused_state_dt;"
        "\n    const double dx_dt = sigma * (-x + y);"
        "\n    values[1] = dx_dt;"
        "\n    const double dy_dt = x * (rho - z) - y;"
        "\n    values[2] = dy_dt;"
        "\n    const double dz_dt = (-beta) * z + x * y;"
        "\n    values[3] = dz_dt;"
        "\n}"
        "\n"
    )
    codegen_remove = CCodeGenerator(ode_unused, remove_unused=True)
    assert codegen_remove.rhs() == (
        "\nvoid rhs(const double t, const double *__restrict states, const double *__restrict parameters, double *values)"
        "\n{"
        "\n"
        "\n    // Assign states"
        "\n    const double x = states[1];"
        "\n    const double y = states[2];"
        "\n    const double z = states[3];"
        "\n"
        "\n    // Assign parameters"
        "\n    const double beta = parameters[1];"
        "\n    const double rho = parameters[2];"
        "\n    const double sigma = parameters[3];"
        "\n"
        "\n    // Assign expressions"
        "\n    const double dunused_state_dt = 0;"
        "\n    values[0] = dunused_state_dt;"
        "\n    const double dx_dt = sigma * (-x + y);"
        "\n    values[1] = dx_dt;"
        "\n    const double dy_dt = x * (rho - z) - y;"
        "\n    values[2] = dy_dt;"
        "\n    const double dz_dt = (-beta) * z + x * y;"
        "\n    values[3] = dz_dt;"
        "\n}"
        "\n"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="clang-format-docs is not available on Windows")
def test_c_remove_unused_forward_explicit_euler(ode_unused):
    codegen_orig = CCodeGenerator(ode_unused)
    assert codegen_orig.scheme(get_scheme("forward_euler")) == (
        "\nvoid forward_euler(const double *__restrict states, const double t, const double dt,"
        "\n                   const double *__restrict parameters, double *values)"
        "\n{"
        "\n"
        "\n    // Assign states"
        "\n    const double unused_state = states[0];"
        "\n    const double x = states[1];"
        "\n    const double y = states[2];"
        "\n    const double z = states[3];"
        "\n"
        "\n    // Assign parameters"
        "\n    const double a = parameters[0];"
        "\n    const double beta = parameters[1];"
        "\n    const double rho = parameters[2];"
        "\n    const double sigma = parameters[3];"
        "\n    const double unused_parameter = parameters[4];"
        "\n"
        "\n    // Assign expressions"
        "\n    const double unused_expression = 0;"
        "\n    const double dunused_state_dt = 0;"
        "\n    values[0] = dt * dunused_state_dt + unused_state;"
        "\n    const double dx_dt = sigma * (-x + y);"
        "\n    values[1] = dt * dx_dt + x;"
        "\n    const double dy_dt = x * (rho - z) - y;"
        "\n    values[2] = dt * dy_dt + y;"
        "\n    const double dz_dt = (-beta) * z + x * y;"
        "\n    values[3] = dt * dz_dt + z;"
        "\n}"
        "\n"
    )
    codegen_remove = CCodeGenerator(ode_unused, remove_unused=True)
    assert codegen_remove.scheme(get_scheme("forward_euler")) == (
        "\nvoid forward_euler(const double *__restrict states, const double t, const double dt,"
        "\n                   const double *__restrict parameters, double *values)"
        "\n{"
        "\n"
        "\n    // Assign states"
        "\n    const double unused_state = states[0];"
        "\n    const double x = states[1];"
        "\n    const double y = states[2];"
        "\n    const double z = states[3];"
        "\n"
        "\n    // Assign parameters"
        "\n    const double beta = parameters[1];"
        "\n    const double rho = parameters[2];"
        "\n    const double sigma = parameters[3];"
        "\n"
        "\n    // Assign expressions"
        "\n    const double dunused_state_dt = 0;"
        "\n    values[0] = dt * dunused_state_dt + unused_state;"
        "\n    const double dx_dt = sigma * (-x + y);"
        "\n    values[1] = dt * dx_dt + x;"
        "\n    const double dy_dt = x * (rho - z) - y;"
        "\n    values[2] = dt * dy_dt + y;"
        "\n    const double dz_dt = (-beta) * z + x * y;"
        "\n    values[3] = dt * dz_dt + z;"
        "\n}"
        "\n"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="clang-format-docs is not available on Windows")
def test_c_remove_unused_forward_generalized_rush_larsen(ode_unused):
    codegen_orig = CCodeGenerator(ode_unused)
    assert codegen_orig.scheme(get_scheme("forward_generalized_rush_larsen")) == (
        "\nvoid forward_generalized_rush_larsen(const double *__restrict states, const double t, const double dt,"
        "\n                                     const double *__restrict parameters, double *values)"
        "\n{"
        "\n"
        "\n    // Assign states"
        "\n    const double unused_state = states[0];"
        "\n    const double x = states[1];"
        "\n    const double y = states[2];"
        "\n    const double z = states[3];"
        "\n"
        "\n    // Assign parameters"
        "\n    const double a = parameters[0];"
        "\n    const double beta = parameters[1];"
        "\n    const double rho = parameters[2];"
        "\n    const double sigma = parameters[3];"
        "\n    const double unused_parameter = parameters[4];"
        "\n"
        "\n    // Assign expressions"
        "\n    const double unused_expression = 0;"
        "\n    const double dunused_state_dt = 0;"
        "\n    values[0] = dt * dunused_state_dt + unused_state;"
        "\n    const double dx_dt = sigma * (-x + y);"
        "\n    const double dx_dt_linearized = -sigma;"
        "\n    values[1] = x + ((fabs(dx_dt_linearized) > 1e-08) ? (dx_dt * (exp(dt * dx_dt_linearized) - 1) / dx_dt_linearized)"
        "\n                                                      : (dt * dx_dt));"
        "\n    const double dy_dt = x * (rho - z) - y;"
        "\n    const double dy_dt_linearized = -1;"
        "\n    values[2] = y + ((fabs(dy_dt_linearized) > 1e-08) ? (dy_dt * (exp(dt * dy_dt_linearized) - 1) / dy_dt_linearized)"
        "\n                                                      : (dt * dy_dt));"
        "\n    const double dz_dt = (-beta) * z + x * y;"
        "\n    const double dz_dt_linearized = -beta;"
        "\n    values[3] = z + ((fabs(dz_dt_linearized) > 1e-08) ? (dz_dt * (exp(dt * dz_dt_linearized) - 1) / dz_dt_linearized)"
        "\n                                                      : (dt * dz_dt));"
        "\n}"
        "\n"
    )
    codegen_remove = CCodeGenerator(ode_unused, remove_unused=True)
    assert codegen_remove.scheme(get_scheme("forward_generalized_rush_larsen")) == (
        "\nvoid forward_generalized_rush_larsen(const double *__restrict states, const double t, const double dt,"
        "\n                                     const double *__restrict parameters, double *values)"
        "\n{"
        "\n"
        "\n    // Assign states"
        "\n    const double unused_state = states[0];"
        "\n    const double x = states[1];"
        "\n    const double y = states[2];"
        "\n    const double z = states[3];"
        "\n"
        "\n    // Assign parameters"
        "\n    const double beta = parameters[1];"
        "\n    const double rho = parameters[2];"
        "\n    const double sigma = parameters[3];"
        "\n"
        "\n    // Assign expressions"
        "\n    const double dunused_state_dt = 0;"
        "\n    values[0] = dt * dunused_state_dt + unused_state;"
        "\n    const double dx_dt = sigma * (-x + y);"
        "\n    const double dx_dt_linearized = -sigma;"
        "\n    values[1] = x + ((fabs(dx_dt_linearized) > 1e-08) ? (dx_dt * (exp(dt * dx_dt_linearized) - 1) / dx_dt_linearized)"
        "\n                                                      : (dt * dx_dt));"
        "\n    const double dy_dt = x * (rho - z) - y;"
        "\n    const double dy_dt_linearized = -1;"
        "\n    values[2] = y + ((fabs(dy_dt_linearized) > 1e-08) ? (dy_dt * (exp(dt * dy_dt_linearized) - 1) / dy_dt_linearized)"
        "\n                                                      : (dt * dy_dt));"
        "\n    const double dz_dt = (-beta) * z + x * y;"
        "\n    const double dz_dt_linearized = -beta;"
        "\n    values[3] = z + ((fabs(dz_dt_linearized) > 1e-08) ? (dz_dt * (exp(dt * dz_dt_linearized) - 1) / dz_dt_linearized)"
        "\n                                                      : (dt * dz_dt));"
        "\n}"
        "\n"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="clang-format-docs is not available on Windows")
def test_c_conditional_expression_assignment(parser, trans):
    expr = """
    states(v=0)
    ah = Conditional(Ge(v, -40), 0, 0.057*exp(-(v + 80)/6.8))
    dv_dt = 0
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)
    ode = make_ode(*result, name="conditional")
    codegen = CCodeGenerator(ode)
    assert codegen.rhs() == (
        "\nvoid rhs(const double t, const double *__restrict states, const double *__restrict parameters, double *values)"
        "\n{"
        "\n"
        "\n    // Assign states"
        "\n    const double v = states[0];"
        "\n"
        "\n    // Assign parameters"
        "\n"
        "\n    // Assign expressions"
        "\n    const double dv_dt = 0;"
        "\n    values[0] = dv_dt;"
        "\n    const double ah = (v >= -40) ? 0 : 0.057 * exp((-(v + 80)) / 6.8);"
        "\n}"
        "\n"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="clang-format-docs is not available on Windows")
def test_c_conditional_expression_advanced(parser, trans):
    expr = """
    states(v=0.5)
    parameters(g=1)
    gammasu  = g*Conditional(Gt(Gt(v,0)*v, Lt(v, -1)*(-v-1)), Gt(v,0)*v, Lt(v, -1)*(-v-1))
    dv_dt = 0
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)
    ode = make_ode(*result, name="conditional")
    codegen = CCodeGenerator(ode)
    assert codegen.rhs() == (
        "\nvoid rhs(const double t, const double *__restrict states, const double *__restrict parameters, double *values)"
        "\n{"
        "\n"
        "\n    // Assign states"
        "\n    const double v = states[0];"
        "\n"
        "\n    // Assign parameters"
        "\n    const double g = parameters[0];"
        "\n"
        "\n    // Assign expressions"
        "\n    const double dv_dt = 0;"
        "\n    values[0] = dv_dt;"
        "\n    const double gammasu ="
        "\n        g * ((((v > 0 && v < -1) ? (v > -v - 1) : (((v > 0) ? (v > 0) : (((v < -1) ? (v > -1) : (0)))))))"
        "\n                 ? (v * ((v > 0) ? (1) : (0)))"
        "\n                 : ((-v - 1) * ((v < -1) ? (1) : (0))));"
        "\n}"
        "\n"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="clang-format-docs is not available on Windows")
def test_c_codegen_parameter_index(codegen: CCodeGenerator):
    assert codegen.parameter_index() == (
        "// Parameter index"
        "\nint parameter_index(const char name[])"
        "\n{"
        "\n"
        '\n    if (strcmp(name, "a") == 0)'
        "\n    {"
        "\n        return 0;"
        "\n    }"
        "\n"
        '\n    else if (strcmp(name, "beta") == 0)'
        "\n    {"
        "\n        return 1;"
        "\n    }"
        "\n"
        '\n    else if (strcmp(name, "rho") == 0)'
        "\n    {"
        "\n        return 2;"
        "\n    }"
        "\n"
        '\n    else if (strcmp(name, "sigma") == 0)'
        "\n    {"
        "\n        return 3;"
        "\n    }"
        "\n"
        "\n    return -1;"
        "\n}"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="clang-format-docs is not available on Windows")
def test_c_codegen_state_index(codegen: CCodeGenerator):
    assert codegen.state_index() == (
        "// State index"
        "\nint state_index(const char name[])"
        "\n{"
        "\n"
        '\n    if (strcmp(name, "x") == 0)'
        "\n    {"
        "\n        return 0;"
        "\n    }"
        "\n"
        '\n    else if (strcmp(name, "y") == 0)'
        "\n    {"
        "\n        return 1;"
        "\n    }"
        "\n"
        '\n    else if (strcmp(name, "z") == 0)'
        "\n    {"
        "\n        return 2;"
        "\n    }"
        "\n"
        "\n    return -1;"
        "\n}"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="clang-format-docs is not available on Windows")
def test_c_codegen_monitor_index(codegen: CCodeGenerator):
    assert codegen.monitor_index() == (
        "// Monitor index"
        "\nint monitor_index(const char name[])"
        "\n{"
        "\n"
        '\n    if (strcmp(name, "dx_dt") == 0)'
        "\n    {"
        "\n        return 0;"
        "\n    }"
        "\n"
        '\n    else if (strcmp(name, "dy_dt") == 0)'
        "\n    {"
        "\n        return 1;"
        "\n    }"
        "\n"
        '\n    else if (strcmp(name, "dz_dt") == 0)'
        "\n    {"
        "\n        return 2;"
        "\n    }"
        "\n"
        "\n    return -1;"
        "\n}"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="clang-format-docs is not available on Windows")
def test_c_codegen_monitor(codegen: CCodeGenerator):
    assert codegen.monitor_values() == (
        "\nvoid monitor_values(const double t, const double *__restrict states, const double *__restrict parameters,"
        "\n                    double *values)"
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
        "\n    values[0] = dx_dt;"
        "\n    const double dy_dt = x * (rho - z) - y;"
        "\n    values[1] = dy_dt;"
        "\n    const double dz_dt = (-beta) * z + x * y;"
        "\n    values[2] = dz_dt;"
        "\n}"
        "\n"
    )
