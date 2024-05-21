# flake8: noqa: E501
import math

import lark
import pytest
import sympy as sp
from gotranx.expressions import build_expression
from gotranx.units import ureg


@pytest.mark.parametrize("expr", ["x=1", "x = 1", "x= 1"])
def test_assignment_single_1(expr, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0].name == "x"
    assert result[0].value.tree == lark.Tree(
        "scientific",
        [lark.Token("SCIENTIFIC_NUMBER", "1")],
    )


@pytest.mark.parametrize(
    "expr",
    [
        "x = 1 + 2",
        "x= 1 +2",
        "x = 1+ 2",
    ],
)
def test_assignment_single_2_terms(expr, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0].name == "x"
    assert result[0].value.tree == lark.Tree(
        lark.Token("RULE", "expression"),
        [
            lark.Tree(lark.Token("RULE", "scientific"), [lark.Token("SCIENTIFIC_NUMBER", "1")]),
            lark.Token("PLUS", "+"),
            lark.Tree(lark.Token("RULE", "scientific"), [lark.Token("SCIENTIFIC_NUMBER", "2")]),
        ],
    )


def test_assignment_single_3_terms(parser, trans):
    tree = parser.parse("x = 1 * 2 + 3")
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0].name == "x"
    assert result[0].value.tree == lark.Tree(
        lark.Token("RULE", "expression"),
        [
            lark.Tree(
                lark.Token("RULE", "term"),
                [
                    lark.Tree(
                        lark.Token("RULE", "scientific"),
                        [lark.Token("SCIENTIFIC_NUMBER", "1")],
                    ),
                    lark.Token("STAR", "*"),
                    lark.Tree(
                        lark.Token("RULE", "scientific"),
                        [lark.Token("SCIENTIFIC_NUMBER", "2")],
                    ),
                ],
            ),
            lark.Token("PLUS", "+"),
            lark.Tree(lark.Token("RULE", "scientific"), [lark.Token("SCIENTIFIC_NUMBER", "3")]),
        ],
    )


def test_assignment_single_4_terms(parser, trans):
    tree = parser.parse("x = 1 * 2 + 3 - 4")
    result = trans.transform(tree)
    assert len(result) == 1

    assert result[0].name == "x"
    assert result[0].value.tree == lark.Tree(
        lark.Token("RULE", "expression"),
        [
            lark.Tree(
                lark.Token("RULE", "term"),
                [
                    lark.Tree(
                        lark.Token("RULE", "scientific"),
                        [lark.Token("SCIENTIFIC_NUMBER", "1")],
                    ),
                    lark.Token("STAR", "*"),
                    lark.Tree(
                        lark.Token("RULE", "scientific"),
                        [lark.Token("SCIENTIFIC_NUMBER", "2")],
                    ),
                ],
            ),
            lark.Token("PLUS", "+"),
            lark.Tree(lark.Token("RULE", "scientific"), [lark.Token("SCIENTIFIC_NUMBER", "3")]),
            lark.Token("NEG", "-"),
            lark.Tree(lark.Token("RULE", "scientific"), [lark.Token("SCIENTIFIC_NUMBER", "4")]),
        ],
    )


@pytest.mark.parametrize(
    "expr",
    [
        "x = 1 * 2 + 3 - (4 / 5)",
        "x = 1 * 2 + 3 - 4 / 5",
        "x = (1 * 2) + 3 - (4 / 5)",
    ],
)
def test_assignment_single5(expr, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0].name == "x"
    assert result[0].value.tree == lark.Tree(
        lark.Token("RULE", "expression"),
        [
            lark.Tree(
                lark.Token("RULE", "term"),
                [
                    lark.Tree(
                        lark.Token("RULE", "scientific"),
                        [lark.Token("SCIENTIFIC_NUMBER", "1")],
                    ),
                    lark.Token("STAR", "*"),
                    lark.Tree(
                        lark.Token("RULE", "scientific"),
                        [lark.Token("SCIENTIFIC_NUMBER", "2")],
                    ),
                ],
            ),
            lark.Token("PLUS", "+"),
            lark.Tree(lark.Token("RULE", "scientific"), [lark.Token("SCIENTIFIC_NUMBER", "3")]),
            lark.Token("NEG", "-"),
            lark.Tree(
                lark.Token("RULE", "term"),
                [
                    lark.Tree(
                        lark.Token("RULE", "scientific"),
                        [lark.Token("SCIENTIFIC_NUMBER", "4")],
                    ),
                    lark.Token("SLASH", "/"),
                    lark.Tree(
                        lark.Token("RULE", "scientific"),
                        [lark.Token("SCIENTIFIC_NUMBER", "5")],
                    ),
                ],
            ),
        ],
    )


def test_assignment_single_with_names(parser, trans):
    expr = "x = (1 * y) + rho - (z / sigma)"
    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 1
    assert result[0].name == "x"
    assert result[0].value.tree == lark.Tree(
        lark.Token("RULE", "expression"),
        [
            lark.Tree(
                lark.Token("RULE", "term"),
                [
                    lark.Tree(
                        lark.Token("RULE", "scientific"),
                        [lark.Token("SCIENTIFIC_NUMBER", "1")],
                    ),
                    lark.Token("STAR", "*"),
                    lark.Tree(lark.Token("RULE", "variable"), [lark.Token("VARIABLE", "y")]),
                ],
            ),
            lark.Token("PLUS", "+"),
            lark.Tree(lark.Token("RULE", "variable"), [lark.Token("VARIABLE", "rho")]),
            lark.Token("NEG", "-"),
            lark.Tree(
                lark.Token("RULE", "term"),
                [
                    lark.Tree(lark.Token("RULE", "variable"), [lark.Token("VARIABLE", "z")]),
                    lark.Token("SLASH", "/"),
                    lark.Tree(
                        lark.Token("RULE", "variable"),
                        [lark.Token("VARIABLE", "sigma")],
                    ),
                ],
            ),
        ],
    )


@pytest.mark.parametrize("expr", ["x=1\ny=2", "x = 1 \n y =2", "x= 1\n y = 2"])
def test_assignment_double(expr, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 2
    assert result[0].name == "x"
    assert result[0].value.tree == lark.Tree(
        "scientific",
        [lark.Token("SCIENTIFIC_NUMBER", "1")],
    )
    assert result[1].name == "y"
    assert result[1].value.tree == lark.Tree(
        "scientific",
        [lark.Token("SCIENTIFIC_NUMBER", "2")],
    )


def test_expressions_with_name_only(parser, trans):
    expr = """
    expressions("My Component")
    x = 1
    y = 2
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)
    components = result.components

    assert len(components) == 1
    component = components[0]
    assert component.name == "My Component"

    assignments = component.assignments
    assert len(assignments) == 2

    x = component.find_assignment("x")
    assert x.name == "x"
    assert x.value.tree == lark.Tree(
        "scientific",
        [lark.Token("SCIENTIFIC_NUMBER", "1")],
    )

    y = component.find_assignment("y")
    assert y.name == "y"
    assert y.value.tree == lark.Tree(
        "scientific",
        [lark.Token("SCIENTIFIC_NUMBER", "2")],
    )


def test_expressions_with_two_components(parser, trans):
    expr = """
    expressions("My Component", "My second component")
    x = 1
    y = 2
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)

    components = result.components

    assert len(components) == 2
    for name, component in zip(["My Component", "My second component"], components):
        assert component.name == name
        assignments = component.assignments
        assert len(assignments) == 2

        x = component.find_assignment("x")
        assert x.components == ("My Component", "My second component")
        assert x.name == "x"
        assert x.value.tree == lark.Tree(
            "scientific",
            [lark.Token("SCIENTIFIC_NUMBER", "1")],
        )

        y = component.find_assignment("y")
        assert x.components == ("My Component", "My second component")
        assert y.name == "y"
        assert y.value.tree == lark.Tree(
            "scientific",
            [lark.Token("SCIENTIFIC_NUMBER", "2")],
        )


def test_expressions_with_name_and_unit(parser, trans):
    expr = """
    expressions("My Component", "My second component")
    x = 1  # mV
    y = 2  # mol
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)

    tree = parser.parse(expr)
    result = trans.transform(tree)

    components = result.components

    assert len(components) == 2
    for name, component in zip(["My Component", "My second component"], components):
        assert component.name == name
        assignments = component.assignments
        assert len(assignments) == 2

        x = component.find_assignment("x")
        assert x.comment is None
        assert x.components == ("My Component", "My second component")
        assert x.name == "x"
        assert x.value.tree == lark.Tree(
            "scientific",
            [lark.Token("SCIENTIFIC_NUMBER", "1")],
        )
        assert x.unit == ureg.Unit("mV")

        y = component.find_assignment("y")
        assert y.components == ("My Component", "My second component")
        assert y.name == "y"
        assert y.value.tree == lark.Tree(
            "scientific",
            [lark.Token("SCIENTIFIC_NUMBER", "2")],
        )
        assert y.unit == ureg.Unit("mol")
        assert y.comment is None


def test_expressions_with_invalid_unit_becomes_comment(parser, trans):
    expr = """
    expressions("My Component", "My second component")
    x = 1  # Some variable that we want to define
    y = 2  # 1 + 1
    z = 2  # 3.14
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)

    components = result.components

    assert len(components) == 2
    for name, component in zip(["My Component", "My second component"], components):
        assert component.name == name
        assignments = component.assignments
        assert len(assignments) == 3

        x = component.find_assignment("x")
        assert x.components == ("My Component", "My second component")
        assert x.name == "x"
        assert x.value.tree == lark.Tree(
            "scientific",
            [lark.Token("SCIENTIFIC_NUMBER", "1")],
        )
        assert x.unit is None
        assert x.comment.text == "Some variable that we want to define"

        y = component.find_assignment("y")
        assert y.components == ("My Component", "My second component")
        assert y.name == "y"
        assert y.value.tree == lark.Tree(
            "scientific",
            [lark.Token("SCIENTIFIC_NUMBER", "2")],
        )
        assert y.unit is None
        assert y.comment.text == "1 + 1"

        z = component.find_assignment("z")
        assert x.components == ("My Component", "My second component")
        assert z.name == "z"
        assert z.value.tree == lark.Tree(
            "scientific",
            [lark.Token("SCIENTIFIC_NUMBER", "2")],
        )
        assert z.unit is None
        assert z.comment.text == "3.14"


@pytest.mark.parametrize(
    "expr, subs, expected",
    [
        ("y = log(x * 2)", {"x": 1}, math.log(2.0)),
        ("y = log(x + log(2))", {"x": 1}, math.log(1 + math.log(2.0))),
        ("y = exp(x * 2)", {"x": 1}, math.exp(2.0)),
        ("y = exp(x + log(2) - 1 )", {"x": 1}, 2),
        ("y = sin(x)", {"x": math.pi / 3}, math.sqrt(3) / 2),
        ("y = cos(x)", {"x": math.pi / 3}, 0.5),
        ("y = tan(x)", {"x": math.pi / 3}, math.sqrt(3)),
        ("y = asin(x)", {"x": math.sqrt(3) / 2}, math.pi / 3),
        ("y = acos(x)", {"x": 0.5}, math.pi / 3),
        ("y = atan(x)", {"x": math.sqrt(3)}, math.pi / 3),
        ("y = sqrt(x)", {"x": 4}, 2),
        ("y = Abs(x)", {"x": -2}, 2),
    ],
)
def test_expression_functions(expr, subs, expected, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    symbols = {name: sp.Symbol(name) for name in subs}
    sympy_expr = build_expression(result[0].value.tree, symbols=symbols)
    assert math.isclose(sympy_expr.subs(subs), expected)


@pytest.mark.parametrize(
    "expr, subs, expected",
    [
        ("\n x = 1\n y = x**2", {"x": 1}, 1),
        ("\n x = 1\n y = x**2", {"x": 2}, 4),
        ("\n x = 1\n y = (1 + x) ** 2", {"x": 2}, 9),
        ("\n x = 1\n y = ((2 + x) / 2) ** (1 + x)", {"x": 2}, 8),
    ],
)
def test_power_expressions(expr, subs, expected, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    symbols = {name: sp.Symbol(name) for name in subs}
    y = result.components[0].find_assignment("y")
    sympy_expr = build_expression(y.value.tree, symbols=symbols)
    assert math.isclose(sympy_expr.subs(subs), expected)


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("\n x = pi", math.pi),
        ("\n x = 3 * pi", 3 * math.pi),
    ],
)
def test_expressions_constants(expr, expected, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    x = result.components[0].find_assignment("x")
    sympy_expr = build_expression(x.value.tree)
    assert math.isclose(sympy_expr, expected)


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("\n x = 1e3", 1000),
        ("\n x = 1e-3", 0.001),
        ("\n x = 1E2", 100),
        ("\n x = 1E-2", 0.01),
        ("\n x = 2e3", 2000),
        ("\n x = 2e-3", 0.002),
        ("\n x = 2E2", 200),
        ("\n x = 2E-2", 0.02),
    ],
)
def test_expressions_scientific_notation(expr, expected, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    x = result.components[0].find_assignment("x")
    sympy_expr = build_expression(x.value.tree)
    assert math.isclose(sympy_expr, expected)


@pytest.mark.parametrize(
    "expr, unit",
    [
        ("x = 1", None),
        ("x = 1 # mV", ureg.Unit("mV")),
        ("x = 1 # pA*pF**-1", ureg.Unit("pA*pF**-1")),
    ],
)
def test_assignment_with_unit(expr, unit, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    # Since expression don't start with a newline we get
    # an Assignment object
    x = result[0]
    assert x.unit == unit


@pytest.mark.parametrize(
    "expr, subs, expected",
    [
        (
            "alpha_h = Conditional(Lt(V, -40), 0.057*exp(-(V + 80)/6.8), +0) # ms**-1",
            {"V": 0},
            0,
        ),
        (
            "alpha_h = Conditional(Lt(V, -40), 0.057*exp(-(V + 80)/6.8), 0) # ms**-1",
            {"V": -80},
            0.057,
        ),
        (
            "alpha_h = Conditional(Gt(V, -40), 0.057*exp(-(V + 80)/6.8), 0) # ms**-1",
            {"V": 0},
            0.057 * math.exp(-80 / 6.8),
        ),
        (
            "alpha_h = Conditional(Gt(V, -40), 0.057*exp(-(V + 80)/6.8), 0) # ms**-1",
            {"V": -80},
            0,
        ),
    ],
)
def test_lt_gt_conditional(expr, subs, expected, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    symbols = {name: sp.Symbol(name) for name in subs}
    alpha_h = result[0]
    sympy_expr = build_expression(alpha_h.value.tree, symbols=symbols)

    assert math.isclose(sympy_expr.subs(subs), expected)


@pytest.mark.parametrize(
    "subs, expected",
    [
        (
            {
                "stim_period": 100,
                "time": 0,
                "stim_amplitude": 42,
                "stim_start": 1,
                "stim_duration": 10,
            },
            0,
        ),
        (
            {
                "stim_period": 100,
                "time": 2,
                "stim_amplitude": 42,
                "stim_start": 1,
                "stim_duration": 10,
            },
            -42,
        ),
        (
            {
                "stim_period": 100,
                "time": 12,
                "stim_amplitude": 42,
                "stim_start": 1,
                "stim_duration": 10,
            },
            0,
        ),
        (
            {
                "stim_period": 100,
                "time": 102,
                "stim_amplitude": 42,
                "stim_start": 1,
                "stim_duration": 10,
            },
            -42,
        ),
    ],
)
def test_stimulus_current(subs, expected, parser, trans):
    expr = """
    i_Stim = Conditional(And(Ge(time - floor(time/stim_period)*stim_period, stim_start),Le(time - floor(time/stim_period)*stim_period, stim_start + stim_duration)),-stim_amplitude, 0) # pA*pF**-1
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)
    i_Stim = result.components[0].find_assignment("i_Stim")
    symbols = {name: sp.Symbol(name) for name in i_Stim.value.dependencies}
    sympy_expr = build_expression(i_Stim.value.tree, symbols=symbols)

    assert math.isclose(sympy_expr.subs(subs), expected)


def test_stimulus_current_with_trailing_comma(parser, trans):
    subs = {
        "stim_period": 100,
        "time": 0,
        "stim_amplitude": 42,
        "stim_start": 1,
        "stim_duration": 10,
    }
    expected = 0
    expr = """
    i_Stim = Conditional(And(Ge(time - floor(time/stim_period)*stim_period, stim_start),Le(time - floor(time/stim_period)*stim_period, stim_start + stim_duration),),-stim_amplitude, 0) # pA*pF**-1
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)
    i_Stim = result.components[0].find_assignment("i_Stim")
    symbols = {name: sp.Symbol(name) for name in i_Stim.value.dependencies}
    sympy_expr = build_expression(i_Stim.value.tree, symbols=symbols)

    assert math.isclose(sympy_expr.subs(subs), expected)


@pytest.mark.parametrize(
    "subs, expected",
    [
        (
            {
                "time": 0,
                "stim_amplitude": 42,
                "stim_start": 1,
                "stim_duration": 10,
            },
            math.exp(-1 / 0.2) * 42,
        ),
        (
            {
                "time": 2,
                "stim_amplitude": 42,
                "stim_start": 1,
                "stim_duration": 10,
            },
            42 - math.exp(-1 / 0.2) * 42,
        ),
        (
            {
                "time": 12,
                "stim_amplitude": 42,
                "stim_start": 1,
                "stim_duration": 10,
            },
            math.exp(-1 / 0.2) * 42,
        ),
    ],
)
def test_stimulus_current_continuous(subs, expected, parser, trans):
    expr = """
    i_Stim = stim_amplitude * ContinuousConditional(Ge(time, stim_start), 1, 0, 0.2) * ContinuousConditional(Le(time, stim_start + stim_duration), 1, 0, 0.2)
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)
    i_Stim = result.components[0].find_assignment("i_Stim")
    symbols = {name: sp.Symbol(name) for name in i_Stim.value.dependencies}
    sympy_expr = build_expression(i_Stim.value.tree, symbols=symbols)
    assert math.isclose(sympy_expr.subs(subs), expected, rel_tol=1e-2)
