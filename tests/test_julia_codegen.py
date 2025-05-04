import pytest
from gotranx.schemes import get_scheme
from gotranx.codegen import JuliaCodeGenerator
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
    rhoz = rho - z
    dy_dt = x*rhoz - y # millivolt
    dx_dt = sigma*(-x + y)
    betaz = beta*z
    dz_dt = -betaz + x*y
    """
    tree = parser.parse(expr)
    return make_ode(*trans.transform(tree), name="lorentz")


@pytest.fixture(scope="module")
def codegen(ode) -> JuliaCodeGenerator:
    return JuliaCodeGenerator(ode)


def test_julia_codegen_initial_state_values(codegen: JuliaCodeGenerator):
    assert codegen.initial_state_values() == (
        "\nfunction init_state_values!(states)"
        "\n    #="
        "\n    x=1.0, z=3.05, y=2.0"
        "\n    =#"
        "\n    states[1] = 1.0"
        "\n    states[2] = 3.05"
        "\n    states[3] = 2.0"
        "\nend"
        "\n"
    )


def test_julia_codegen_parameter_index(codegen: JuliaCodeGenerator):
    assert codegen.parameter_index() == (
        "# Parameter index"
        "\nfunction parameter_index(name::String)"
        "\n"
        '\n    if name == "a"'
        "\n        return 1"
        "\n"
        "\n"
        '\n    elseif name == "beta"'
        "\n        return 2"
        "\n"
        "\n"
        '\n    elseif name == "rho"'
        "\n        return 3"
        "\n"
        "\n"
        '\n    elseif name == "sigma"'
        "\n        return 4"
        "\n"
        "\n    end"
        "\n    return -1"
        "\nend"
    )


def test_julia_codegen_initial_parameter_values(codegen: JuliaCodeGenerator):
    assert codegen.initial_parameter_values() == (
        "\nfunction init_parameter_values!(parameters)"
        "\n    #="
        "\n    a=0, beta=2.4, rho=21.0, sigma=12.0"
        "\n    =#"
        "\n    parameters[1] = 0"
        "\n    parameters[2] = 2.4"
        "\n    parameters[3] = 21.0"
        "\n    parameters[4] = 12.0"
        "\nend"
        "\n"
    )


@pytest.mark.parametrize(
    "order, arguments",
    [
        (
            RHSArgument.stp,
            ("states, t, parameters"),
        ),
        (
            RHSArgument.spt,
            ("states, parameters, t"),
        ),
        (
            RHSArgument.tsp,
            ("t, states, parameters"),
        ),
    ],
)
def test_julia_codegen_rhs(order: str, arguments: str, codegen: JuliaCodeGenerator):
    assert codegen.rhs(order=order) == (
        f"\nfunction rhs({arguments}, values)"
        "\n"
        "\n    # Assign states"
        "\n    x = states[1]"
        "\n    z = states[2]"
        "\n    y = states[3]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[1]"
        "\n    beta = parameters[2]"
        "\n    rho = parameters[3]"
        "\n    sigma = parameters[4]"
        "\n"
        "\n    # Assign expressions"
        "\n    betaz = beta .* z"
        "\n    rhoz = rho - z"
        "\n    dx_dt = sigma .* (-x + y)"
        "\n    values[1] = dx_dt"
        "\n    dz_dt = -betaz + x .* y"
        "\n    values[2] = dz_dt"
        "\n    dy_dt = rhoz .* x - y"
        "\n    values[3] = dy_dt"
        "\nend"
        "\n"
    )


def test_julia_codegen_explicit_euler(codegen: JuliaCodeGenerator):
    assert codegen.scheme(get_scheme("explicit_euler")) == (
        "\nfunction explicit_euler(states, t, dt, parameters, values)"
        "\n"
        "\n    # Assign states"
        "\n    x = states[1]"
        "\n    z = states[2]"
        "\n    y = states[3]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[1]"
        "\n    beta = parameters[2]"
        "\n    rho = parameters[3]"
        "\n    sigma = parameters[4]"
        "\n"
        "\n    # Assign expressions"
        "\n    betaz = beta .* z"
        "\n    rhoz = rho - z"
        "\n    dx_dt = sigma .* (-x + y)"
        "\n    values[1] = dt .* dx_dt + x"
        "\n    dz_dt = -betaz + x .* y"
        "\n    values[2] = dt .* dz_dt + z"
        "\n    dy_dt = rhoz .* x - y"
        "\n    values[3] = dt .* dy_dt + y"
        "\nend"
        "\n"
    )


def test_julia_monitored(codegen: JuliaCodeGenerator):
    assert codegen.monitor_values() == (
        "\nfunction monitor_values(t, states, parameters, values)"
        "\n"
        "\n    # Assign states"
        "\n    x = states[1]"
        "\n    z = states[2]"
        "\n    y = states[3]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[1]"
        "\n    beta = parameters[2]"
        "\n    rho = parameters[3]"
        "\n    sigma = parameters[4]"
        "\n"
        "\n    # Assign expressions"
        "\n    betaz = beta .* z"
        "\n    values[1] = betaz"
        "\n    rhoz = rho - z"
        "\n    values[2] = rhoz"
        "\n    dx_dt = sigma .* (-x + y)"
        "\n    values[3] = dx_dt"
        "\n    dz_dt = -betaz + x .* y"
        "\n    values[4] = dz_dt"
        "\n    dy_dt = rhoz .* x - y"
        "\n    values[5] = dy_dt"
        "\nend"
        "\n"
    )


def test_consistent_floats(parser, trans):
    expr = """
    \nstates(x=0)
    \ndx_dt = Conditional(Ge(x, 31.4978), 1.0, 1.0763*exp(-1.007*exp(-0.0829*x)))
    """

    tree = parser.parse(expr)
    result = trans.transform(tree)
    ode = make_ode(*result, name="name")
    codegen = JuliaCodeGenerator(ode)
    rhs = codegen.rhs()

    assert rhs == (
        "\nfunction rhs(t, states, parameters, values)"
        "\n"
        "\n    # Assign states"
        "\n    x = states[1]"
        "\n"
        "\n    # Assign parameters"
        "\n"
        "\n"
        "\n    # Assign expressions"
        "\n    dx_dt = ((x >= 31.4978) ? (1.0) : (1.0763 * exp((-1.007) * exp((-0.0829) * x))))"
        "\n    values[1] = dx_dt"
        "\nend"
        "\n"
    )


def test_consistent_floats_with_T(parser, trans):
    expr = """
    \nstates(x=0)
    \ndx_dt = Conditional(Ge(x, 31.4978), 1.0, 1.0763*exp(-1.007*exp(-0.0829*x)))
    """

    tree = parser.parse(expr)
    result = trans.transform(tree)
    ode = make_ode(*result, name="name")
    codegen = JuliaCodeGenerator(ode, type_stable=True)
    rhs = codegen.rhs()
    assert rhs == (
        "\nfunction rhs(t::TYPE, states::AbstractVector{TYPE}, parameters::AbstractVector{TYPE}, values::AbstractVector{TYPE}) where {TYPE}"  # noqa: E501
        "\n"
        "\n    # Assign states"
        "\n    x = states[1]"
        "\n"
        "\n    # Assign parameters"
        "\n"
        "\n"
        "\n    # Assign expressions"
        "\n    dx_dt = ((x >= TYPE(31.4978)) ? (TYPE(1.0)) : (TYPE(1.0763) * exp((-TYPE(1.007)) * exp((-TYPE(0.0829)) * x))))"  # noqa: E501
        "\n    values[1] = dx_dt"
        "\nend"
        "\n"
    )
