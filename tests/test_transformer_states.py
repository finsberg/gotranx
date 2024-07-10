import pytest
from gotranx import atoms


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
    assert result[0] == atoms.State(name="x", value=1)


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

    assert result[0] == atoms.State(name="x", value=1)
    assert result[1] == atoms.State(name="y", value=2)


def test_states_with_component(parser, trans):
    expr = 'states("My component", x=1, y=2)'
    tree = parser.parse(expr)
    result = trans.transform(tree)

    assert len(result) == 2

    assert result[0] == atoms.State(name="x", value=1, components=("My component",))
    assert result[1] == atoms.State(name="y", value=2, components=("My component",))


def test_states_with_component_and_info(parser, trans):
    expr = 'states("My component", "Some info about the component", x=1, y=2)'
    tree = parser.parse(expr)
    result = trans.transform(tree)

    assert len(result) == 2

    assert result[0] == atoms.State(
        name="x",
        value=1,
        components=(
            "My component",
            "Some info about the component",
        ),
    )
    assert result[1] == atoms.State(
        name="y",
        value=2,
        components=(
            "My component",
            "Some info about the component",
        ),
    )


def test_different_sets_of_states(parser, trans):
    expr = (
        'states("First component", "Some info about first '
        'component", x=1, y=2)\nstates("Second component", z=3)\nstates(w=4)'
    )
    tree = parser.parse(expr)
    result = trans.transform(tree).components

    assert len(result) == 4
    first_component = result[0]

    assert first_component.states == result[1].states
    assert first_component.name == "First component"
    assert first_component.states == {
        atoms.State(
            name="x",
            value=1,
            components=(
                "First component",
                "Some info about first component",
            ),
        ),
        atoms.State(
            name="y",
            value=2,
            components=(
                "First component",
                "Some info about first component",
            ),
        ),
    }

    second_component = result[2]
    assert second_component.name == "Second component"
    assert second_component.states == {
        atoms.State(name="z", value=3, components=("Second component",)),
    }

    third_component = result[3]
    assert third_component.name == ""
    assert third_component.states == {
        atoms.State(name="w", value=4, components=("",)),
    }


@pytest.mark.parametrize(
    "expr, expected",
    [
        (
            'states(x=ScalarParam(1, unit="pA"))',
            (atoms.State(name="x", value=1, unit_str="pA"),),
        ),
        (
            'states(x=ScalarParam(1, unit="pA", description="Info about x"))',
            (
                atoms.State(
                    name="x",
                    value=1,
                    unit_str="pA",
                    description="Info about x",
                ),
            ),
        ),
        (
            'states(x=ScalarParam(1, unit="pA*pF**-1", description="Info about x"))',
            (
                atoms.State(
                    name="x",
                    value=1,
                    unit_str="pA*pF**-1",
                    description="Info about x",
                ),
            ),
        ),
        (
            """states(
            x=ScalarParam(1, unit="pA", description="Info about x"),
            y=ScalarParam(2.0, unit="mM", description="Info about y")
        )""",
            (
                atoms.State(
                    name="x",
                    value=1,
                    unit_str="pA",
                    description="Info about x",
                ),
                atoms.State(
                    name="y",
                    value=2.0,
                    unit_str="mM",
                    description="Info about y",
                ),
            ),
        ),
    ],
)
def test_states_with_unit_and_desc(expr, expected, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert result == expected
