import pytest
from gotran_parser import atoms


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
