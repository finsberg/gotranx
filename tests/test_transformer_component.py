import lark
from gotran_parser import atoms


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
            rhs=atoms.Expression(lark.Tree("number", [lark.Token("NUMBER", "0")])),
        ),
        atoms.Assignment(
            lhs="db_dt",
            rhs=atoms.Expression(lark.Tree("number", [lark.Token("NUMBER", "1")])),
        ),
        atoms.Assignment(
            lhs="dc_dt",
            rhs=atoms.Expression(lark.Tree("number", [lark.Token("NUMBER", "2")])),
        ),
    }
