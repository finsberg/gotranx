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


def test_parameters_with_component(parser, trans):
    expr = 'parameters("My Component", x=1, y=2)'
    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 2
    assert result[0] == atoms.Parameter(name="x", value=1, component="My Component")
    assert result[1] == atoms.Parameter(name="y", value=2, component="My Component")


def test_three_sets_of_parameters_with_component(parser, trans):
    expr = 'parameters("First component", x=1, y=2)\nparameters("Second component", z=3)\nparameters(w=4)'
    tree = parser.parse(expr)
    result = trans.transform(tree)

    assert len(result) == 3
    first_component = result[0]
    assert first_component.name == "First component"
    assert first_component.parameters == {
        atoms.Parameter(name="x", value=1, component="First component"),
        atoms.Parameter(name="y", value=2, component="First component"),
    }

    second_component = result[1]
    assert second_component.name == "Second component"
    assert second_component.parameters == {
        atoms.Parameter(name="z", value=3, component="Second component"),
    }

    third_component = result[2]
    assert third_component.name is None
    assert third_component.parameters == {
        atoms.Parameter(name="w", value=4, component=None),
    }
