import lark
import pytest
from gotran_parser import atoms
from gotran_parser import exceptions
from gotran_parser import ode_component


def test_component_None(parser, trans):
    tree = parser.parse(
        "parameters(x=1, y=2)\nparameters(z=3)\nstates(a=1)\nstates(b=2, c=3)\nda_dt=0\ndb_dt=1\ndc_dt=2",
    )
    result = trans.transform(tree).components
    assert len(result) == 1  # Only one component
    comp = result[0]
    assert comp.name is None
    assert comp.parameters == {
        atoms.Parameter(name="x", value=1),
        atoms.Parameter(name="y", value=2),
        atoms.Parameter(name="z", value=3),
    }
    assert comp.states == {
        atoms.State(name="a", value=1),
        atoms.State(name="b", value=2),
        atoms.State(name="c", value=3),
    }

    assert comp.assignments == {
        atoms.Assignment(
            name="da_dt",
            value=atoms.Expression(
                tree=lark.Tree("scientific", [lark.Token("SCIENTIFIC_NUMBER", "0")]),
            ),
        ),
        atoms.Assignment(
            name="db_dt",
            value=atoms.Expression(
                tree=lark.Tree("scientific", [lark.Token("SCIENTIFIC_NUMBER", "1")]),
            ),
        ),
        atoms.Assignment(
            name="dc_dt",
            value=atoms.Expression(
                tree=lark.Tree("scientific", [lark.Token("SCIENTIFIC_NUMBER", "2")]),
            ),
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
    comp = result.components[0]

    assert comp.intermediates == {
        atoms.Intermediate(
            name="c",
            value=atoms.Expression(
                tree=lark.Tree(
                    "add",
                    [
                        lark.Tree("variable", [lark.Token("VARIABLE", "a")]),
                        lark.Tree("variable", [lark.Token("VARIABLE", "b")]),
                    ],
                ),
            ),
            component=None,
        ),
    }

    assert comp.state_derivatives == {
        atoms.StateDerivative(
            name="da_dt",
            value=atoms.Expression(
                tree=lark.Tree("scientific", [lark.Token("SCIENTIFIC_NUMBER", "0")]),
            ),
            component=None,
            state=atoms.State(name="a", value=2.0, component=None, info=None),
        ),
        atoms.StateDerivative(
            name="db_dt",
            value=atoms.Expression(
                tree=lark.Tree(
                    "sub",
                    [
                        lark.Tree("variable", [lark.Token("VARIABLE", "c")]),
                        lark.Tree("variable", [lark.Token("VARIABLE", "a")]),
                    ],
                ),
            ),
            component=None,
            state=atoms.State(name="b", value=3.0, component=None, info=None),
        ),
    }


@pytest.mark.parametrize(
    "expr, is_complete",
    [("states(a=2, b=3)\n da_dt=0\n db_dt=c - a", True), ("states(x=1)\n y=x", False)],
)
def test_component_is_complete(expr, is_complete, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    comp = result.components[0]
    assert comp.is_complete() is is_complete


def test_StateNotFound(parser, trans):
    expr = "states(x=1)\n y=x \n dy_dt=0"
    tree = parser.parse(expr)

    with pytest.raises(exceptions.StateNotFoundInComponent) as e:
        trans.transform(tree)

    assert str(e.value) == "State with name 'y' not found in component None"


def test_state_with_and_without_derivatives(parser, trans):
    expr = "states(x=1, y=2)\n dy_dt=0"
    tree = parser.parse(expr)
    result = trans.transform(tree)
    comp = result.components[0]
    assert comp.states_with_derivatives == {
        atoms.State(name="y", value=2, component=None, info=None),
    }
    assert comp.states_without_derivatives == {
        atoms.State(name="x", value=1, component=None, info=None),
    }
    assert comp.is_complete() is False


@pytest.mark.parametrize(
    "expr, is_match, state_name",
    [
        ("dx_dt", True, "x"),
        ("dx_y_dt", True, "x_y"),
        ("dAb_x_t_dt", True, "Ab_x_t"),
        ("dx_dt_dt", True, "x_dt"),
        ("dxdt", False, ""),
        ("Dx_dt", False, ""),
        ("a_dx_dt", False, ""),
        ("adx_dt", False, ""),
        ("x_dt", False, ""),
        ("dx_dt_", False, ""),
        ("_dx_dt", False, ""),
        ("dx_dts", False, ""),
    ],
)
def test_STATE_DERIV_EXPR(expr, is_match, state_name):
    result = ode_component.STATE_DERIV_EXPR.match(expr)
    if is_match:
        assert result.groupdict()["state"] == state_name
    else:
        assert result is None
