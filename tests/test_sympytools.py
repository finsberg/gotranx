import pytest
import sympy
from gotranx.ode import make_ode
from gotranx.ode import ODE
from gotranx import sympytools


@pytest.fixture(scope="module")
def ode(trans, parser) -> ODE:
    expr = """
    parameters(a=0)
    parameters("My component",
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0,
    beta=2.4
    )
    states("My component", x=1.0, z=3.05, y=2.0)

    expressions("My component")
    dx_dt = sigma*(-x + y)
    dy_dt = y_int - y # millivolt
    dz_dt = z_int + x*y
    y_int = x*(rho - z)
    z_int = -beta*z
    """
    tree = parser.parse(expr)
    return make_ode(*trans.transform(tree))


# @pytest.fixture(scope="module")
# def sym_ode(ode: ODE) -> SympyODE:
#     return SympyODE(ode)


def test_states(ode):
    states = sympytools.states_matrix(ode)
    assert len(states) == 3
    assert str(states[0]) == "x"
    assert str(states[1]) == "y"
    assert str(states[2]) == "z"


def test_Conditional_boolean_condition():
    assert sympytools.Conditional(True, 1, 2) == 1


def test_Conditional_invalid_condition():
    with pytest.raises(TypeError):
        sympytools.Conditional(sympy.Eq, 1, 2)


def test_ContinuousConditional_invalid_condition():
    with pytest.raises(TypeError):
        sympytools.ContinuousConditional(sympy.Eq, 1, 2)


def test_rhs_matrix(ode: ODE):
    rhs = sympytools.rhs_matrix(ode)
    assert len(rhs) == 3
    assert str(rhs[0]) == "sigma*(-x + y)"
    assert str(rhs[1]) == "x*(rho - z) - y"
    assert str(rhs[2]) == "(-beta)*z + x*y"


def test_rhs_matrix_does_not_distribute_coefficients(trans, parser):
    # rhs_matrix eliminates intermediates via xreplace(). Without wrapping
    # that call in `with sympy.core.parameters.evaluate(False)`, sympy
    # rebuilds every ancestor of a substituted symbol using the default
    # (evaluate=True) constructor, which auto-distributes numeric
    # coefficients over sums - e.g. (tm - 4.823)/51.12 would silently
    # become 0.0195618153364632*tm - 0.0943466353677621 once tm is
    # substituted with an expression containing v. Matches the same fix
    # applied in myokit.gotran_to_myokit.
    expr = """
    states(v=-91.33918)
    dv_dt = (tm - 1*4.823)/51.12
    tm = v
    """
    tree = parser.parse(expr)
    ode = make_ode(*trans.transform(tree))
    rhs = sympytools.rhs_matrix(ode)
    assert str(rhs[0]) == "(v - 4.823)/51.12"


def test_jacobian_matrix(ode: ODE):
    jac = sympytools.jacobi_matrix(ode)

    assert str(jac[0]) == "-sigma"
    assert str(jac[1]) == "sigma"
    assert str(jac[2]) == "0"

    assert str(jac[3]) == "rho - z"
    assert str(jac[4]) == "-1"
    assert str(jac[5]) == "-x"

    assert str(jac[6]) == "y"
    assert str(jac[7]) == "x"
    assert str(jac[8]) == "-beta"


def _piecewise_pair():
    """Two guarded expressions that share a subexpression.

    This is the shape a model produces once two assignments are guarded, and
    it is the shape that provokes the hoist. A single Piecewise on its own
    does not: `cse` shares the whole subtree and leaves its interior alone.
    """
    import sympy

    V = sympy.Symbol("V", real=True)
    inner = (V + 40) / (sympy.exp(-(V + 40) / 10) - 1)
    return V, [
        sympy.Piecewise((sympy.Integer(0), V < -40), (inner, True)),
        sympy.Piecewise((sympy.Integer(1), V < -40), (inner * 2, True)),
    ]


def _hoisted_out_of_a_branch(replacements):
    import sympy

    return [
        sub_expr
        for _, sub_expr in replacements
        if sub_expr.has(sympy.exp) and not sub_expr.has(sympy.Piecewise)
    ]


def test_cse_hoists_out_of_a_piecewise_branch():
    """Pins the sympy behaviour that `cse_hiding_piecewise` exists to avoid.

    Plain `sympy.cse` pulls `(V + 40)/(1 - exp(-V/10 - 4))` out of the
    Piecewise branches below and computes it unconditionally. That temporary
    is nan at exactly V = -40 -- 0/0 -- even though neither branch that uses
    it is taken there.
    """
    import numpy
    import sympy

    V, exprs = _piecewise_pair()
    replacements, _ = sympy.cse(exprs, optimizations="basic")
    hoisted = _hoisted_out_of_a_branch(replacements)
    assert hoisted, "expected plain sympy.cse to hoist out of the Piecewise"

    numpy.seterr(all="ignore")
    at_the_boundary = [
        float(sympy.lambdify(V, sub_expr, "numpy")(numpy.float64(-40.0)))
        for sub_expr in hoisted
    ]
    assert not all(numpy.isfinite(at_the_boundary)), at_the_boundary


def test_cse_hiding_piecewise_does_not_hoist_out_of_a_branch():
    import sympy

    from gotranx import sympytools

    V, exprs = _piecewise_pair()
    replacements, reduced = sympytools.cse_hiding_piecewise(exprs, optimizations="basic")
    assert _hoisted_out_of_a_branch(replacements) == []

    restored = list(reduced)
    for symbol, sub_expr in reversed(replacements):
        restored = [expr.xreplace({symbol: sub_expr}) for expr in restored]
    for got, want in zip(restored, exprs):
        assert sympy.simplify(got - want) == 0


def test_cse_hiding_piecewise_still_shares_whole_guards():
    """Holding a Piecewise opaque must not stop cse from sharing the whole
    subtree between two expressions -- that is where most of the saving is."""
    import sympy

    from gotranx import sympytools

    V = sympy.Symbol("V", real=True)
    guarded = sympy.Piecewise((sympy.Integer(1), sympy.Abs(V) < 0.01), (1 / V, True))
    replacements, _ = sympytools.cse_hiding_piecewise([guarded * 2, guarded * 3])
    assert any(sub_expr == guarded for _, sub_expr in replacements)
