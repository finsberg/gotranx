import lark
import pytest
from gotran_parser import atoms
from gotran_parser import exceptions


def test_component_None(parser, trans):
    tree = parser.parse(
        "parameters(x=1, y=2)\nparameters(z=3)\nstates(a=1)\nstates(b=2, c=3)\nda_dt=0\ndb_dt=1\ndc_dt=2",
    )
    result = trans.transform(tree)
    assert len(result) == 1  # Only one component
    comp = result[0]
    assert comp.name is None
    assert comp.parameters == {
        atoms.Parameter(name="x", value=1),
        atoms.Parameter(name="y", value=2),
        atoms.Parameter(name="z", value=3),
    }
    assert comp.states == {
        atoms.State(name="a", ic=1),
        atoms.State(name="b", ic=2),
        atoms.State(name="c", ic=3),
    }

    assert comp.assignments == {
        atoms.Assignment(
            lhs="da_dt",
            rhs=atoms.Expression(tree=lark.Tree("number", [lark.Token("NUMBER", "0")])),
        ),
        atoms.Assignment(
            lhs="db_dt",
            rhs=atoms.Expression(tree=lark.Tree("number", [lark.Token("NUMBER", "1")])),
        ),
        atoms.Assignment(
            lhs="dc_dt",
            rhs=atoms.Expression(tree=lark.Tree("number", [lark.Token("NUMBER", "2")])),
        ),
    }


def test_component_intermediates(parser, trans):
    tree = parser.parse(
        (
            "parameters(x=1, y=2)\n"
            "states(a=2, b=3)\n"
            "da_dt=0\n"
            "c=a+b\n"
            "db_dt=c - a"
        ),
    )
    result = trans.transform(tree)
    comp = result[0]

    assert comp.intermediates == {
        atoms.Intermediate(
            lhs="c",
            rhs=atoms.Expression(
                tree=lark.Tree(
                    "add",
                    [
                        lark.Tree("name", [lark.Token("NAME", "a")]),
                        lark.Tree("name", [lark.Token("NAME", "b")]),
                    ],
                ),
            ),
            component=None,
        ),
    }

    assert comp.state_derivatives == {
        atoms.StateDerivative(
            lhs="da_dt",
            rhs=atoms.Expression(tree=lark.Tree("number", [lark.Token("NUMBER", "0")])),
            component=None,
            state=atoms.State(name="a", ic=2.0, component=None, info=None),
        ),
        atoms.StateDerivative(
            lhs="db_dt",
            rhs=atoms.Expression(
                tree=lark.Tree(
                    "sub",
                    [
                        lark.Tree("name", [lark.Token("NAME", "c")]),
                        lark.Tree("name", [lark.Token("NAME", "a")]),
                    ],
                ),
            ),
            component=None,
            state=atoms.State(name="b", ic=3.0, component=None, info=None),
        ),
    }


@pytest.mark.parametrize(
    "expr, is_complete",
    [("states(a=2, b=3)\n da_dt=0\n db_dt=c - a", True), ("states(x=1)\n y=x", False)],
)
def test_component_is_complete(expr, is_complete, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    comp = result[0]
    assert comp.is_complete() is is_complete


def test_StateNotFound(parser, trans):
    expr = "states(x=1)\n y=x \n dy_dt=0"
    tree = parser.parse(expr)

    with pytest.raises(exceptions.StateNotFound) as e:
        trans.transform(tree)

    assert str(e.value) == "State with name 'y' not found"
