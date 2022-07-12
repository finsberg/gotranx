import lark
import pytest


@pytest.mark.parametrize("expr", ["x=1", "x = 1", "x= 1"])
def test_assignment_single_1(expr, parser, trans):

    tree = parser.parse(expr)
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


@pytest.mark.parametrize("expr", ["x=1\ny=2", "x = 1 \n y =2", "x= 1\n y = 2"])
def test_assignment_double(expr, parser, trans):

    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 2
    assert result[0].lhs == "x"
    assert result[0].rhs.tree == lark.Tree("number", [lark.Token("NUMBER", "1")])
    assert result[1].lhs == "y"
    assert result[1].rhs.tree == lark.Tree("number", [lark.Token("NUMBER", "2")])


def test_expressions_with_name(parser, trans):
    expr = """
    expressions("My Component")
    x = 1
    y = 2
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)

    assert len(result) == 2
    assert result[0].component == "My Component"
    assert result[0].lhs == "x"
    assert result[0].rhs.tree == lark.Tree("number", [lark.Token("NUMBER", "1")])
    assert result[1].component == "My Component"
    assert result[1].lhs == "y"
    assert result[1].rhs.tree == lark.Tree("number", [lark.Token("NUMBER", "2")])
