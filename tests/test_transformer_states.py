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


def test_states_with_component(parser, trans):
    expr = 'states("My component", x=1, y=2)'
    tree = parser.parse(expr)
    result = trans.transform(tree)

    assert len(result) == 2

    assert result[0] == atoms.State(name="x", ic=1, component="My component")
    assert result[1] == atoms.State(name="y", ic=2, component="My component")


def test_states_with_component_and_info(parser, trans):
    expr = 'states("My component", "Some info about the component", x=1, y=2)'
    tree = parser.parse(expr)
    result = trans.transform(tree)

    assert len(result) == 2

    assert result[0] == atoms.State(
        name="x",
        ic=1,
        component="My component",
        info="Some info about the component",
    )
    assert result[1] == atoms.State(
        name="y",
        ic=2,
        component="My component",
        info="Some info about the component",
    )


def test_different_sets_of_states(parser, trans):
    expr = 'states("First component", "Some info about first component", x=1, y=2)\nstates("Second component", z=3)\nstates(w=4)'
    tree = parser.parse(expr)
    result = trans.transform(tree)

    assert len(result) == 3
    first_component = result[0]
    assert first_component.name == "First component"
    assert first_component.states == {
        atoms.State(
            name="x",
            ic=1,
            component="First component",
            info="Some info about first component",
        ),
        atoms.State(
            name="y",
            ic=2,
            component="First component",
            info="Some info about first component",
        ),
    }

    second_component = result[1]
    assert second_component.name == "Second component"
    assert second_component.states == {
        atoms.State(name="z", ic=3, component="Second component"),
    }

    third_component = result[2]
    assert third_component.name is None
    assert third_component.states == {
        atoms.State(name="w", ic=4, component=None),
    }
