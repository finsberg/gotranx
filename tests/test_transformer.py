import lark
import pytest
from gotran_parser import Parser
from gotran_parser import transformer
from gotran_parser import TreeToODE


@pytest.fixture(scope="module")
def parser() -> Parser:
    return Parser(parser="lalr", debug=True)


@pytest.fixture(scope="module")
def trans() -> TreeToODE:
    return TreeToODE()


def test_parameters_single(parser, trans):

    tree = parser.parse("parameters(x=1)")
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0] == transformer.Parameter(name="x", value=1)


def test_parameters_double(parser, trans):
    tree = parser.parse("parameters(x=1, y=2)")
    result = trans.transform(tree)
    assert len(result) == 2
    assert result[0] == transformer.Parameter(name="x", value=1)
    assert result[1] == transformer.Parameter(name="y", value=2)


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
    assert result[0] == transformer.State(name="x", ic=1)


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
    assert result[0] == transformer.State(name="x", ic=1)
    assert result[1] == transformer.State(name="y", ic=2)


@pytest.mark.parametrize("expr", ["x=1", "x = 1", "x= 1"])
def test_assignment_single(expr, parser, trans):

    tree = parser.parse("x = 1")
    result = trans.transform(tree)
    assert result.lhs == "x"
    assert result.rhs == lark.Tree("number", [lark.Token("NUMBER", "1")])


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
    assert result.lhs == "x"
    assert result.rhs == lark.Tree(
        "add",
        [
            lark.Tree("number", [lark.Token("NUMBER", "1")]),
            lark.Tree("number", [lark.Token("NUMBER", "2")]),
        ],
    )


def test_assignment_single_3_terms(parser, trans):

    tree = parser.parse("x = 1 * 2 + 3")
    result = trans.transform(tree)
    assert result.lhs == "x"
    assert result.rhs == lark.Tree(
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
    assert result.lhs == "x"
    assert result.rhs == lark.Tree(
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
    assert result.lhs == "x"
    assert result.rhs == lark.Tree(
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
