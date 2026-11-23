"""Second-order Rush-Larsen schemes: exactness, structure, order of convergence."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import sympy

import gotranx
from gotranx import schemes
from gotranx.schemes import Scheme

HERE = Path(__file__).parent
ODEFILES = HERE / "odefiles"


def _model(ode, scheme, **kwargs) -> dict:
    code = gotranx.cli.gotran2py.get_code(ode, scheme=[scheme], **kwargs)
    ns: dict = {}
    exec(compile(code, "<generated>", "exec"), ns)
    return ns


def _run(step, y0, p, T, dt, *extra):
    y, t = np.array(y0, dtype=float), 0.0
    for _ in range(round(T / dt)):
        y = step(y, t, dt, p, *extra)
        t += dt
    return y


def _rates(errors):
    e = np.asarray(errors)
    return np.log2(e[:-1] / e[1:])


def _reference(rhs, y0, p, edges):
    from scipy.integrate import solve_ivp

    y = np.array(y0, dtype=float)
    for a, b in zip(edges[:-1], edges[1:]):
        sol = solve_ivp(
            lambda t, y: rhs(t, y, p), (a, b), y, method="Radau", rtol=1e-12, atol=1e-12
        )
        y = sol.y[:, -1]
    return y


def _lhs_names(eqs):
    names = []
    for e in eqs:
        m = re.match(r"\s*([A-Za-z_]\w*)\s*=", e)
        if m:
            names.append(m.group(1))
    return names


def test_generalized_rush_larsen_2_is_listed():
    assert "generalized_rush_larsen_2" in schemes.list_schemes()
    assert schemes.get_scheme("generalized_rush_larsen_2") is schemes.generalized_rush_larsen_2


def test_every_name_is_assigned_once():
    ode = gotranx.load_ode(ODEFILES / "beeler_reuter_1977.ode")
    names = _lhs_names(schemes.generalized_rush_larsen_2(ode, sympy.Symbol("dt")))
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates


@pytest.mark.parametrize("cse", [True, False])
def test_linear_gate_with_frozen_coefficients_is_exact(cse):
    ode = gotranx.load.ode_from_string(
        "parameters(yinf=0.7, tau=0.3)\nstates(y=0.1)\ndy_dt = (yinf - y)/tau\n"
    )
    m = _model(ode, Scheme.generalized_rush_larsen_2, cse=cse)
    dt = 0.25
    out = m["generalized_rush_larsen_2"](
        m["init_state_values"](), 0.0, dt, m["init_parameter_values"]()
    )
    assert out[0] == pytest.approx(0.7 + (0.1 - 0.7) * np.exp(-dt / 0.3), rel=1e-13)


def test_name_collisions_with_generated_names_are_avoided():
    text = """
    parameters(k=2.0)
    states(V=0.3, w=0.2)
    V_mid = 0.5*V
    _linearization_temp_0 = exp(-V)
    _linearization_mid_temp_0 = exp(-w)
    dV_dt = -k*V_mid*_linearization_temp_0 + w
    dw_dt = (_linearization_mid_temp_0 - w)/(1 + V**2)
    """
    renamed = (
        text.replace("V_mid", "half_V")
        .replace("_linearization_temp_0", "eV")
        .replace("_linearization_mid_temp_0", "ew")
    )
    a = _model(gotranx.load.ode_from_string(text), Scheme.generalized_rush_larsen_2)
    b = _model(gotranx.load.ode_from_string(renamed), Scheme.generalized_rush_larsen_2)
    y, p = a["init_state_values"](), a["init_parameter_values"]()
    np.testing.assert_allclose(
        a["generalized_rush_larsen_2"](y, 0.1, 0.05, p),
        b["generalized_rush_larsen_2"](y, 0.1, 0.05, p),
        rtol=1e-14,
    )


def test_vectorized_states_match_column_by_column():
    ode = gotranx.load_ode(ODEFILES / "stiff_gate.ode")
    m = _model(ode, Scheme.generalized_rush_larsen_2)
    p = m["init_parameter_values"]()
    y = np.array([[0.1, -0.3, 0.8], [0.0, 0.5, 0.9]])
    out = m["generalized_rush_larsen_2"](y, 0.2, 0.01, p)
    for k in range(y.shape[1]):
        np.testing.assert_allclose(
            out[:, k], m["generalized_rush_larsen_2"](y[:, k].copy(), 0.2, 0.01, p), rtol=1e-14
        )


def test_missing_variables_are_held_constant(parser, trans):
    main = gotranx.ode.make_ode(
        *trans.transform(
            parser.parse(
                """
                parameters("Main", rho=21.0, sigma=12.0)
                parameters("Z", beta=2.4)
                states("Main", x=1.0, y=2.0)
                states("Z", z=3.05)
                expressions("Main")
                dy_dt = x*(rho - z) - y
                dx_dt = sigma*(-x + y)
                expressions("Z")
                dz_dt = -beta*z + x*y
                """
            )
        ),
        name="lorentz",
    )
    z_ode = main.get_component("Z").to_ode()
    m = _model(z_ode, Scheme.generalized_rush_larsen_2)
    x, y, beta, z0, dt = 1.0, 2.0, 2.4, 3.05, 0.3
    out = m["generalized_rush_larsen_2"](
        np.array([z0]), 0.0, dt, m["init_parameter_values"](), np.array([x, y])
    )
    zinf = x * y / beta
    assert out[0] == pytest.approx(zinf + (z0 - zinf) * np.exp(-beta * dt), rel=1e-13)


def test_remove_unused_keeps_indices_consistent(ode_unused):
    dt = 0.01
    full = _model(ode_unused, Scheme.generalized_rush_larsen_2)
    lean = _model(ode_unused, Scheme.generalized_rush_larsen_2, remove_unused=True)
    y, p = full["init_state_values"](), full["init_parameter_values"]()
    np.testing.assert_allclose(
        full["generalized_rush_larsen_2"](y, 0.0, dt, p),
        lean["generalized_rush_larsen_2"](y, 0.0, dt, p),
        rtol=1e-14,
    )


def test_stiff_gate_order_generalized_rush_larsen_2():
    ode = gotranx.load_ode(ODEFILES / "stiff_gate.ode")
    m = _model(ode, Scheme.generalized_rush_larsen_2)
    y0, p, T = m["init_state_values"](), m["init_parameter_values"](), 2.0
    ref = _reference(m["rhs"], y0, p, [0.0, T])
    errors = [
        np.max(np.abs(_run(m["generalized_rush_larsen_2"], y0, p, T, T / 2**k) - ref))
        for k in range(6, 10)
    ]
    rates = _rates(errors)
    assert min(rates[-2:]) >= 1.9, rates


def test_julia_code_is_generated():
    ode = gotranx.load_ode(ODEFILES / "stiff_gate.ode")
    code = gotranx.cli.gotran2julia.get_code(ode, scheme=[Scheme.generalized_rush_larsen_2])
    assert "function generalized_rush_larsen_2(" in code
    assert "g_mid" in code
