import pytest
from gotranx import Parser
from gotranx import TreeToODE
from gotranx.ode import make_ode


@pytest.fixture(scope="module")
def parser() -> Parser:
    return Parser(parser="lalr", debug=True)


@pytest.fixture(scope="module")
def trans() -> TreeToODE:
    return TreeToODE()


@pytest.fixture(scope="module")
def ode_unused(parser, trans):
    expr = """
    parameters(a=0)
    parameters("My component",
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0,
    beta=2.4,
    unused_parameter=1.0
    )
    states("My component", x=1.0, y=2.0,z=3.05, unused_state=1.0)

    expressions("My component")
    dunused_state_dt = 0
    unused_expression = 0
    dy_dt = x*(rho - z) - y # millivolt
    dx_dt = sigma*(-x + y)
    dz_dt = -beta*z + x*y
    """
    tree = parser.parse(expr)
    return make_ode(*trans.transform(tree))
