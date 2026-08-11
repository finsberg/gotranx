import pytest
from gotranx.ode import make_ode
from gotranx.codegen.markdown import MarkdownGenerator
from gotranx.templates import markdown


@pytest.fixture(scope="module")
def ode(trans, parser):
    expr = """
    parameters(a=1.0)
    parameters("My Component", sigma=12.0, rho=21.0, beta=2.4)
    states("My Component", x=1.0, y=2.0, z=3.05)

    expressions("My Component")
    rhoz = rho - z
    dy_dt = x*rhoz - y
    dx_dt = sigma*(-x + y)
    betaz = beta*z
    dz_dt = -betaz + x*y
    """
    tree = parser.parse(expr)
    return make_ode(*trans.transform(tree), name="lorentz")


def test_parameter_table():
    params = [
        {"name": "a", "value": "1.0", "unit": "-", "description": "-"},
        {"name": "b", "value": "2.0", "unit": "mV", "description": "desc"},
    ]
    table = markdown.parameter_table(params)
    assert "| Name | Value | Unit | Description |" in table
    assert "| :--- | :--- | :--- | :--- |" in table
    assert "| a | 1.0 | - | - |" in table
    assert "| b | 2.0 | mV | desc |" in table


def test_state_table():
    states = [
        {"name": "x", "value": "1.0", "unit": "mM", "description": "desc"},
    ]
    table = markdown.state_table(states)
    assert "| Name | Initial Value | Unit | Description |" in table
    assert "| x | 1.0 | mM | desc |" in table


def test_equation_block():
    eqs = ["a = b + c", "d = e * f"]
    block = markdown.equation_block(eqs)
    assert "$$" in block
    assert "\\begin{aligned}" in block
    assert "a = b + c" in block
    assert "d = e * f" in block
    assert "\\end{aligned}" in block


def test_markdown_codegen(ode):
    codegen = MarkdownGenerator(ode)
    markdown_text = codegen.generate()

    # Check title
    assert "# lorentz" in markdown_text

    # Check Component
    assert "## Component: My Component" in markdown_text

    # Check Parameters in table
    assert "| `sigma` | $12.0$ | - | - |" in markdown_text
    assert "| `rho` | $21.0$ | - | - |" in markdown_text
    assert "| `beta` | $2.4$ | - | - |" in markdown_text

    # Check States in table
    assert "| `x` | $1.0$ | - | - |" in markdown_text
    assert "| `y` | $2.0$ | - | - |" in markdown_text
    assert "| `z` | $3.05$ | - | - |" in markdown_text

    # Check Equations
    # Note: Sympy printing might vary slightly, but key parts should be there
    assert "\\frac{d x}{dt} &= \\sigma \\cdot \\left(- x + y\\right) \\\\" in markdown_text
    assert "\\frac{d y}{dt} &= rhoz \\cdot x - y \\\\" in markdown_text
    assert "\\frac{d z}{dt} &= - betaz + x \\cdot y \\\\" in markdown_text
    assert "rhoz &= \\rho - z \\\\" in markdown_text
    assert "betaz &= \\beta \\cdot z \\\\" in markdown_text
