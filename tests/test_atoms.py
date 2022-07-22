import lark
import pytest
import sympy as sp
from gotran_parser import atoms


@pytest.mark.parametrize(
    "expr, deps",
    [
        ("x = (1 * y) + rho - (z / sigma)", {"y", "rho", "z", "sigma"}),
        ("x = (1 * y) + rho - (y / 2)", {"y", "rho"}),
        ("x = 1", set()),
    ],
)
def test_expression_dependencies(expr, deps, parser, trans):

    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert result[0].value.dependencies == deps


def test_parameter_arguments(parser, trans):

    expr = """parameters("My component",
        y = 1.0,
        x = ScalarParam(42.0),
        w = ScalarParam(10.2, unit="mM"),
        v = ScalarParam(3.14, unit="mV", description="Description of v"),
        z = ScalarParam(42.0, description="Description of z")
    )
    """

    tree = parser.parse(expr)
    result = trans.transform(tree)

    assert result[0] == atoms.Parameter(name="y", value=1.0, component="My component")
    assert result[1] == atoms.Parameter(name="x", value=42.0, component="My component")
    assert result[2] == atoms.Parameter(
        name="w",
        value=10.2,
        component="My component",
        unit_str="mM",
    )
    assert result[3] == atoms.Parameter(
        name="v",
        value=3.14,
        description="Description of v",
        component="My component",
        unit_str="mV",
    )
    assert result[4] == atoms.Parameter(
        name="z",
        value=42.0,
        description="Description of z",
        component="My component",
    )


def test_states_arguments(parser, trans):

    expr = """
    states("My component",
        y = 1.0,
        x = ScalarParam(42.0)
    )
    states("My component", "Some info",
        w = ScalarParam(10.2, unit="mM"),
        v = ScalarParam(3.14, unit="mV", description="Description of v")
    )
    states("My other component", "other info",
        z = ScalarParam(42.0, description="Description of z")
    )
    """

    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result.components) == 2

    comp1 = result.components[0]
    assert comp1.name == "My component"
    assert comp1.states == {
        atoms.State(name="y", value=1.0, component="My component"),
        atoms.State(name="x", value=42.0, component="My component"),
        atoms.State(
            name="w",
            value=10.2,
            component="My component",
            unit_str="mM",
            info="Some info",
        ),
        atoms.State(
            name="v",
            value=3.14,
            description="Description of v",
            component="My component",
            info="Some info",
            unit_str="mV",
        ),
    }

    comp2 = result.components[1]
    assert comp2.name == "My other component"
    assert comp2.states == {
        atoms.State(
            name="z",
            value=42.0,
            description="Description of z",
            component="My other component",
            info="other info",
        ),
    }


def test_comment(parser, trans):
    expr = """
    # This is one comment.
    # Here is another comment
    x = 1
    parameters(y = 2)
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)

    assert len(result.comments) == 1
    assert result.comments[0] == atoms.Comment(
        text="This is one comment. Here is another comment",
    )

    comp = result.components
    assert len(comp) == 1
    assert comp[0].parameters == {atoms.Parameter(name="y", value=2.0)}

    assert comp[0].assignments == {
        atoms.Assignment(
            name="x",
            value=atoms.Expression(
                tree=lark.Tree("scientific", [lark.Token("SCIENTIFIC_NUMBER", "1")]),
            ),
        ),
    }


def test_TimeDependentState(parser, trans):
    expr = """
    states(x = ScalarParam(1, unit="M"))
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)
    t = sp.Symbol("t")
    x = result[0]
    x_t = x.to_TimeDependentState(t)

    assert str(x.symbol) == "x"
    assert str(x_t.symbol) == "x(t)"


# def test_StateDerivative(parser, trans):
#     expr = """
#     parameters(a = 1)
#     states(x=2)
#     dx_dt = a + 1
#     """
#     tree = parser.parse(expr)
#     result = trans.transform(tree)
#     assert len(result) == 1
#     comp = result[0]
#     assert len(comp.state_derivatives) == 1
#     dx_dt = comp.state_derivatives.pop()
#     t = sp.Symbol("t")
#     breakpoint()
