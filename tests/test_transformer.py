import lark
import pytest
from gotran_parser import atoms


def test_parameters_single(parser, trans):

    tree = parser.parse("parameters(x=1)")
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0] == atoms.Parameter(name="x", value=1)


def test_parameters_double(parser, trans):
    tree = parser.parse("parameters(x=1, y=2)")
    result = trans.transform(tree)
    assert len(result) == 2
    assert result[0] == atoms.Parameter(name="x", value=1)
    assert result[1] == atoms.Parameter(name="y", value=2)


def test_ode_component_None(parser, trans):
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


@pytest.mark.parametrize(
    "expr",
    [
        "states(x=1)",
        "states(x = 1)",
        "states(x= 1)",
        """states(
    x=1)""",
        """states(
    x=1
    )""",
    ],
)
def test_states_single(expr, parser, trans):

    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0] == atoms.State(name="x", ic=1)


@pytest.mark.parametrize(
    "expr",
    [
        "states(x=1, y=2)",
        "states(x = 1, y =2)",
        "states(x= 1,y= 2)",
        """states(
    x=1, y=2)""",
        """states(
    x=1,
    y=2
    )""",
    ],
)
def test_states_double(expr, parser, trans):

    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 2

    assert result[0] == atoms.State(name="x", ic=1)
    assert result[1] == atoms.State(name="y", ic=2)


@pytest.mark.parametrize("expr", ["x=1", "x = 1", "x= 1"])
def test_assignment_single(expr, parser, trans):

    tree = parser.parse("x = 1")
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0].lhs == "x"
    assert result[0].rhs.tree == lark.Tree("number", [lark.Token("NUMBER", "1")])


@pytest.mark.parametrize(
    "expr",
    [
        "x = 1 + 2",
        "x= 1 +2",
        "x = 1+ 2",
        """x= 1
+ 2""",
    ],
)
def test_assignment_single_2_terms(expr, parser, trans):

    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0].lhs == "x"
    assert result[0].rhs.tree == lark.Tree(
        "add",
        [
            lark.Tree("number", [lark.Token("NUMBER", "1")]),
            lark.Tree("number", [lark.Token("NUMBER", "2")]),
        ],
    )


def test_assignment_single_3_terms(parser, trans):

    tree = parser.parse("x = 1 * 2 + 3")
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0].lhs == "x"
    assert result[0].rhs.tree == lark.Tree(
        "add",
        [
            lark.Tree(
                "mul",
                [
                    lark.Tree("number", [lark.Token("NUMBER", "1")]),
                    lark.Tree("number", [lark.Token("NUMBER", "2")]),
                ],
            ),
            lark.Tree("number", [lark.Token("NUMBER", "3")]),
        ],
    )


def test_assignment_single_4_terms(parser, trans):

    tree = parser.parse("x = 1 * 2 + 3 - 4")
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0].lhs == "x"
    assert result[0].rhs.tree == lark.Tree(
        "sub",
        [
            lark.Tree(
                "add",
                [
                    lark.Tree(
                        "mul",
                        [
                            lark.Tree("number", [lark.Token("NUMBER", "1")]),
                            lark.Tree("number", [lark.Token("NUMBER", "2")]),
                        ],
                    ),
                    lark.Tree("number", [lark.Token("NUMBER", "3")]),
                ],
            ),
            lark.Tree("number", [lark.Token("NUMBER", "4")]),
        ],
    )


@pytest.mark.parametrize(
    "expr",
    [
        "x = 1 * 2 + 3 - (4 / 5)",
        "x = 1 * 2 + 3 - 4 / 5",
        "x = (1 * 2) + 3 - (4 / 5)",
        "x=(1 * 2 +3) - (4 / 5)",
    ],
)
def test_assignment_single5(expr, parser, trans):

    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0].lhs == "x"
    assert result[0].rhs.tree == lark.Tree(
        "sub",
        [
            lark.Tree(
                "add",
                [
                    lark.Tree(
                        "mul",
                        [
                            lark.Tree("number", [lark.Token("NUMBER", "1")]),
                            lark.Tree("number", [lark.Token("NUMBER", "2")]),
                        ],
                    ),
                    lark.Tree("number", [lark.Token("NUMBER", "3")]),
                ],
            ),
            lark.Tree(
                "div",
                [
                    lark.Tree("number", [lark.Token("NUMBER", "4")]),
                    lark.Tree("number", [lark.Token("NUMBER", "5")]),
                ],
            ),
        ],
    )


def test_assignment_single_with_names(parser, trans):

    expr = "x = (1 * y) + rho - (z / sigma)"
    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0].lhs == "x"
    assert result[0].rhs.tree == lark.Tree(
        "sub",
        [
            lark.Tree(
                "add",
                [
                    lark.Tree(
                        "mul",
                        [
                            lark.Tree("number", [lark.Token("NUMBER", "1")]),
                            lark.Tree("name", [lark.Token("NAME", "y")]),
                        ],
                    ),
                    lark.Tree("name", [lark.Token("NAME", "rho")]),
                ],
            ),
            lark.Tree(
                "div",
                [
                    lark.Tree("name", [lark.Token("NAME", "z")]),
                    lark.Tree("name", [lark.Token("NAME", "sigma")]),
                ],
            ),
        ],
    )
