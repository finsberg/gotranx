import pytest
from gotran_parser import atoms


@pytest.mark.parametrize(
    "expr",
    [
        "parameters(x=1)",
        "parameters(x = 1)",
        "parameters(x= 1)",
        """parameters(
    x=1)""",
        """parameters(
    x=1
    )""",
    ],
)
def test_parameters_single(expr, parser, trans):

    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0] == atoms.Parameter(name="x", value=1)


@pytest.mark.parametrize(
    "expr",
    [
        "parameters(x=1, y=2)",
        "parameters(x = 1, y =2)",
        "parameters(x= 1,y= 2)",
        """parameters(
    x=1, y=2)""",
        """parameters(
    x=1,
    y=2
    )""",
    ],
)
def test_parameters_double(expr, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 2
    assert result[0] == atoms.Parameter(name="x", value=1)
    assert result[1] == atoms.Parameter(name="y", value=2)
