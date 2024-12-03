import lark
import pytest
import sympy as sp
from gotranx import atoms
from gotranx.ode import gather_atoms


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
    components = trans.transform(tree).components
    result = list(sorted(components[0].parameters, key=lambda x: x.name))

    assert result[0] == atoms.Parameter(
        name="v",
        value=sp.sympify(3.14),
        description="Description of v",
        components=("My component",),
        unit_str="mV",
    )
    assert result[1] == atoms.Parameter(
        name="w",
        value=sp.sympify(10.2),
        components=("My component",),
        unit_str="mM",
    )
    assert result[2] == atoms.Parameter(
        name="x",
        value=sp.sympify(42.0),
        components=("My component",),
    )
    assert result[3] == atoms.Parameter(
        name="y",
        value=sp.sympify(1.0),
        components=("My component",),
    )
    assert result[4] == atoms.Parameter(
        name="z",
        value=sp.sympify(42.0),
        description="Description of z",
        components=("My component",),
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
    assert len(result.components) == 4

    comp1 = result.components[0]
    assert comp1.name == "My component"

    assert comp1.states == {
        atoms.State(name="y", value=sp.sympify(1.0), components=("My component",)),
        atoms.State(name="x", value=sp.sympify(42.0), components=("My component",)),
        atoms.State(
            name="w",
            value=sp.sympify(10.2),
            components=("My component", "Some info"),
            unit_str="mM",
        ),
        atoms.State(
            name="v",
            value=sp.sympify(3.14),
            description="Description of v",
            components=("My component", "Some info"),
            unit_str="mV",
        ),
    }

    comp2 = result.components[2]
    assert comp2.name == "My other component"
    assert comp2.states == {
        atoms.State(
            name="z",
            value=sp.sympify(42.0),
            description="Description of z",
            components=("My other component", "other info"),
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

    assert len(result.comments) == 2
    assert result.comments[0] == atoms.Comment(
        text="This is one comment.",
    )
    assert result.comments[1] == atoms.Comment(
        text="Here is another comment",
    )

    comp = result.components
    assert len(comp) == 1
    assert comp[0].parameters == {atoms.Parameter(name="y", value=sp.sympify(2))}

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
    x = list(result.components[0].states)[0]
    x_t = x.to_TimeDependentState(t)

    assert str(x.symbol) == "x"
    assert str(x_t.symbol) == "x(t)"


def test_stateful_assignments(parser, trans):
    expr = """
    parameters(
    a = 1.0,
    b = 2.0
    )
    states(
    x = 1.0
    )

    z = x /(exp(x) - 1.0)
    w = a + b
    v = x + z
    g = v * a
    """

    tree = parser.parse(expr)
    result = trans.transform(tree)
    comp = result.components[0]

    all_atoms = gather_atoms(components=[comp])

    for state in comp.states:
        assert state.is_stateful(all_atoms.lookup)

    for parameter in comp.parameters:
        assert not parameter.is_stateful(all_atoms.lookup)

    z = comp.find_assignment("z")
    assert z.is_stateful(all_atoms.lookup)

    w = comp.find_assignment("w")
    assert not w.is_stateful(all_atoms.lookup)

    v = comp.find_assignment("v")
    assert v.is_stateful(all_atoms.lookup)

    g = comp.find_assignment("g")
    assert g.is_stateful(all_atoms.lookup)


def test_singularities(parser, trans):
    expr = """
    parameters(
    a = 1.0,
    b = 2.0
    )
    states(
    x = 1.0
    )

    z = x /(exp(x) - 1.0)
    z1 = x /(exp(x) - 1.0) + a
    z2 = x /(exp(x) - 1.0) + a + (x - 2)/(exp(x) - exp(2))
    w = a / b
    v = x / b
    g = b / x
    """

    tree = parser.parse(expr)
    result = trans.transform(tree)
    comp = result.components[0]

    all_atoms = gather_atoms(components=[comp])
    a = all_atoms.symbols["a"]

    z = comp.find_assignment("z")
    z_sing = z.resolve_expression(all_atoms.symbols).singularities(all_atoms.lookup)
    assert len(z_sing) == 1
    assert tuple(z_sing)[0] == atoms.Singularity(
        symbol=all_atoms.symbols["x"], value=0, replacement=1
    )
    assert not tuple(z_sing)[0].is_infinite

    z1 = comp.find_assignment("z1")
    z1_sing = z1.resolve_expression(all_atoms.symbols).singularities(all_atoms.lookup)
    assert len(z1_sing) == 1
    assert tuple(z1_sing)[0] == atoms.Singularity(
        symbol=all_atoms.symbols["x"], value=0, replacement=a + 1
    )
    assert not tuple(z1_sing)[0].is_infinite

    z2 = comp.find_assignment("z2")
    z2_sing = z2.resolve_expression(all_atoms.symbols).singularities(all_atoms.lookup)
    assert len(z2_sing) == 2
    assert z2_sing == frozenset(
        {
            atoms.Singularity(
                symbol=all_atoms.symbols["x"],
                value=2,
                replacement=(-a * sp.exp(2) + a * sp.exp(4) - 1 + 3 * sp.exp(2))
                / (-sp.exp(2) + sp.exp(4)),
            ),
            atoms.Singularity(
                symbol=all_atoms.symbols["x"],
                value=0,
                replacement=(-a + a * sp.exp(2) + 1 + sp.exp(2)) / (-1 + sp.exp(2)),
            ),
        }
    )

    w = comp.find_assignment("w")
    w_sing = w.resolve_expression(all_atoms.symbols).singularities(all_atoms.lookup)
    assert len(w_sing) == 0  # No singularities since b is not stateful

    v = comp.find_assignment("v")
    v_sing = v.resolve_expression(all_atoms.symbols).singularities(all_atoms.lookup)
    assert len(v_sing) == 0  # No singularities since b is not stateful

    g = comp.find_assignment("g")
    g_sing = g.resolve_expression(all_atoms.symbols).singularities(all_atoms.lookup)
    assert len(g_sing) == 1
    assert tuple(g_sing)[0] == atoms.Singularity(
        symbol=all_atoms.symbols["x"], value=0, replacement=sp.oo * sp.sign(all_atoms.symbols["b"])
    )
    assert tuple(g_sing)[0].is_infinite
