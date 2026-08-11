import pytest
from gotranx import atoms


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
    parameters = list(result.components[0].parameters)
    assert len(parameters) == 1
    assert parameters[0] == atoms.Parameter(name="x", value=1)


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
    parameters = sorted(result.components[0].parameters, key=lambda x: x.name)
    assert len(parameters) == 2
    assert parameters[0] == atoms.Parameter(name="x", value=1)
    assert parameters[1] == atoms.Parameter(name="y", value=2)


def test_parameters_with_component(parser, trans):
    expr = 'parameters("My Component", x=1, y=2)'
    tree = parser.parse(expr)
    result = trans.transform(tree)
    parameters = sorted(result.components[0].parameters, key=lambda x: x.name)
    assert len(parameters) == 2
    assert parameters[0] == atoms.Parameter(name="x", value=1, components=("My Component",))
    assert parameters[1] == atoms.Parameter(name="y", value=2, components=("My Component",))


def test_different_sets_of_parameters(parser, trans):
    expr = (
        'parameters("First component", x=1, y=2)\n'
        'parameters("Second component", z=3)\nparameters(w=4)'
    )
    tree = parser.parse(expr)
    result = trans.transform(tree).components

    assert len(result) == 3
    first_component = result[0]
    assert first_component.name == "First component"
    assert first_component.parameters == {
        atoms.Parameter(name="x", value=1, components=("First component",)),
        atoms.Parameter(name="y", value=2, components=("First component",)),
    }

    second_component = result[1]
    assert second_component.name == "Second component"
    assert second_component.parameters == {
        atoms.Parameter(name="z", value=3, components=("Second component",)),
    }

    third_component = result[2]
    assert third_component.name == ""
    assert third_component.parameters == {
        atoms.Parameter(name="w", value=4, components=("",)),
    }


@pytest.mark.parametrize(
    "expr, expected",
    [
        (
            'parameters(x=ScalarParam(1, unit="pA"))',
            (atoms.Parameter(name="x", value=1, unit_str="pA"),),
        ),
        (
            'parameters(x=ScalarParam(1, unit="pA", description="Info about x"))',
            (
                atoms.Parameter(
                    name="x",
                    value=1,
                    unit_str="pA",
                    description="Info about x",
                ),
            ),
        ),
        (
            'parameters(x=ScalarParam(1, unit="pA*pF**-1", description="Info about x"))',
            (
                atoms.Parameter(
                    name="x",
                    value=1,
                    unit_str="pA*pF**-1",
                    description="Info about x",
                ),
            ),
        ),
        (
            """parameters(
            x=ScalarParam(1, unit="pA", description="Info about x"),
            y=ScalarParam(2.0, unit="mM", description="Info about y")
        )""",
            (
                atoms.Parameter(
                    name="x",
                    value=1,
                    unit_str="pA",
                    description="Info about x",
                ),
                atoms.Parameter(
                    name="y",
                    value=2.0,
                    unit_str="mM",
                    description="Info about y",
                ),
            ),
        ),
    ],
)
def test_parameter_with_unit_and_desc(expr, expected, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    parameters = tuple(sorted(result.components[0].parameters, key=lambda x: x.name))
    assert parameters == expected


@pytest.mark.parametrize(
    "expr",
    [
        "parameters(x=1,)",
        "parameters(x = 1 , )",
        """parameters(
            x=1,
        )""",
    ],
)
def test_parameters_single_trailing_comma(expr, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    parameters = list(result.components[0].parameters)
    assert len(parameters) == 1
    assert parameters[0] == atoms.Parameter(name="x", value=1)


@pytest.mark.parametrize(
    "expr",
    [
        "parameters(x=1, y=2,)",
        "parameters(x=1, y=2 , )",
        """parameters(
            x=1,
            y=2,
        )""",
    ],
)
def test_parameters_double_trailing_comma(expr, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    parameters = sorted(result.components[0].parameters, key=lambda x: x.name)
    assert len(parameters) == 2
    assert parameters[0] == atoms.Parameter(name="x", value=1)
    assert parameters[1] == atoms.Parameter(name="y", value=2)
