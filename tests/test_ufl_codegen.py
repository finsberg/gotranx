import math
import pytest
from pathlib import Path
import gotranx
from gotranx.cli import gotran2py, gotran2ufl
from gotranx.codegen.ufl import UFLCodeGenerator
from gotranx.schemes import Scheme

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


def _generated(module, ode, scheme=None):
    code = module.get_code(ode, scheme=scheme)
    namespace: dict = {}
    exec(compile(code, f"<{module.__name__}>", "exec"), namespace)
    return namespace


def test_ufl_generalized_rush_larsen_with_float_parameters():
    """The generated UFL scheme accepts parameters that are plain Python floats.

    The linearized rate of ``x`` is the parameter ``-k``, so the scheme's
    zero-division guard ``|dx_dt_linearized| > delta`` compares two numbers.
    Printed as ``>`` that is a Python ``bool``, which ``ufl.Or`` refuses.
    """
    pytest.importorskip("ufl")
    from ufl.core.expr import Expr

    ode = gotranx.load.ode_from_string(
        """
        parameters(k=2.0, c=1.0)
        states(x=0.2)
        dx_dt = c - k * x
        """
    )
    scheme = [Scheme.generalized_rush_larsen]
    ufl_code = _generated(gotran2ufl, ode, scheme)
    py_code = _generated(gotran2py, ode, scheme)

    states = py_code["init_state_values"]()
    parameters = py_code["init_parameter_values"]()
    dt = 0.1
    expected = py_code["generalized_rush_larsen"](states, 0.0, dt, parameters)

    values = ufl_code["generalized_rush_larsen"](
        [float(s) for s in states], 0.0, dt, [float(p) for p in parameters]
    )

    assert isinstance(values[0], Expr)
    assert float(values[0]) == pytest.approx(expected[0], rel=1e-12)
    # The scheme is exact for a linear ODE: x(dt) = c/k + (x0 - c/k) exp(-k dt)
    assert float(values[0]) == pytest.approx(0.5 - 0.3 * math.exp(-0.2), rel=1e-12)


@pytest.mark.parametrize("b", [1.0, 2.0])
@pytest.mark.parametrize(
    "condition", ["Lt(a, b)", "Le(a, b)", "Gt(a, b)", "Ge(a, b)", "Eq(a, b)", "Not(Eq(a, b))"]
)
def test_ufl_relational_with_float_operands(condition, b):
    """Every relational prints to a UFL condition, whatever its operands are.

    Comparing two Python numbers gives a Python ``bool``, which
    ``ufl.conditional`` refuses. ``==`` and ``!=`` never give a UFL condition,
    even on UFL operands, since UFL reserves them for structural equality.
    """
    pytest.importorskip("ufl")

    ode = gotranx.load.ode_from_string(
        f"""
        parameters(a=1.0, b=2.0)
        states(x=0.0)
        dx_dt = Conditional({condition}, 1.0, -1.0)
        """
    )
    ufl_code = _generated(gotran2ufl, ode)
    py_code = _generated(gotran2py, ode)

    states = py_code["init_state_values"]()
    parameters = py_code["init_parameter_values"](b=b)
    expected = py_code["rhs"](0.0, states, parameters)

    values = ufl_code["rhs"](0.0, [float(s) for s in states], [float(p) for p in parameters])

    assert float(values[0]) == expected[0]
