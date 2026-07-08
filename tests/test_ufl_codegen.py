import pytest
from pathlib import Path
import gotranx
from gotranx.codegen.ufl import UFLCodeGenerator

here = Path(__file__).parent.absolute()


@pytest.fixture
def ode_for_ufl(parser, trans):
    expr = """
    parameters(a=1.5, b=2.0)
    states(x=0.0, y=1.0)

    dx_dt = a * x * exp(b)

    dy_dt = Conditional(And(Gt(x, 0.0), Lt(y, 1.0)), 1.0, 0.0)
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)
    # make_ode builds the full ODE object from the transformed components
    return gotranx.ode.make_ode(*result, name="TestODE")


def test_ufl_codegen_imports(ode_for_ufl):
    codegen = UFLCodeGenerator(ode_for_ufl)
    code = codegen.imports()

    assert "import ufl" in code
    assert "import numpy" in code


def test_ufl_codegen_rhs(ode_for_ufl):
    codegen = UFLCodeGenerator(ode_for_ufl)
    code = codegen.rhs()

    # Verify flattening of values (so they are valid python variable names)
    assert "_values_0 =" in code
    assert "_values_1 =" in code

    # Verify return block is a list (stripping whitespace to ignore black/ruff formatting)
    compact_code = code.replace(" ", "").replace("\n", "")
    assert "return[_values_0,_values_1" in compact_code

    # Verify UFL math replacements
    assert "ufl.exp(b)" in code

    # Verify piecewise conditions map to ufl.conditional and logic maps to ufl.And
    assert "ufl.conditional(" in code
    assert "ufl.And(" in code


def test_ufl_codegen_init_states(ode_for_ufl):
    codegen = UFLCodeGenerator(ode_for_ufl)
    code = codegen.initial_state_values()

    # Should maintain standard numpy arrays for state initializations
    assert "states = numpy.array([0.0, 1.0], dtype=numpy.float64)" in code
    assert "states[state_index(key)] = value" in code


def test_ufl_codegen_full(ode_for_ufl):
    codegen = UFLCodeGenerator(ode_for_ufl)

    # gotranx constructs the final file by concatenating the individual parts
    code = "\n".join(
        [
            codegen.imports(),
            codegen.initial_state_values(),
            codegen.initial_parameter_values(),
            codegen.rhs(),
        ]
    )

    # Check that all the parts are assembled correctly
    assert "import ufl" in code
    assert "def init_state_values" in code
    assert "def init_parameter_values" in code
    assert "def rhs" in code
    assert "ufl.conditional" in code


def test_ufl_codegen_ordmm_land():
    """Stress test UFL generation against a complex, real-world cell model."""
    ode_path = here / "odefiles" / "ORdmm_Land.ode"

    # Load the ODE directly from the file
    ode = gotranx.load_ode(ode_path)

    codegen = UFLCodeGenerator(ode)

    code = "\n".join(
        [
            codegen.imports(),
            codegen.initial_state_values(),
            codegen.initial_parameter_values(),
            codegen.rhs(),
        ]
    )

    # Ensure the standard structure is successfully generated
    assert "import ufl" in code
    assert "import numpy" in code
    assert "def init_state_values(**values):" in code
    assert "def init_parameter_values(**values):" in code
    assert "def rhs(t, states, parameters):" in code

    # The ORdmm_Land model is highly complex and should trigger mathematical conversions
    assert "ufl.exp" in code
    assert "ufl.conditional" in code

    # Ensure we return a flat list containing all the derivatives/states for the ODE
    compact_code = code.replace(" ", "").replace("\n", "")
    assert "return[" in compact_code

    # Check that known states from ORdmm_Land were
    # generated successfully as state array index mappings
    assert "hL = states[" in code
    assert "a = states[" in code
