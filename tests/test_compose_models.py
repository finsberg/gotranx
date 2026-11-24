"""Composition on real models: TT06 + Land 2017 (one way), ToR-ORd + Land 2017 (two way).

The hand-made splits in simcardemsx name the same things differently: ``XS``/``XW`` there
are ``S``/``W`` here, ``Zetas``/``Zetaw`` are ``Zs``/``Zw``, and ``CaTrpn`` is ``CaTRPN``.
"""

from pathlib import Path

import numpy as np
import pytest
from conftest import only, rest

import gotranx
import gotranx.cli.gotran2py
from gotranx.schemes import Scheme

FILES = Path(__file__).parent / "compose_files"


def names(atoms) -> set[str]:
    return {a.name for a in atoms}


@pytest.fixture(scope="module")
def torord_land() -> gotranx.ODE:
    return gotranx.compose(FILES / "torord_land_recipe.toml")


def test_tt06_land_one_way():
    ode = gotranx.compose(FILES / "tt06_land_recipe.toml")
    assert set(only(ode, "mechanics").missing_variables) == {"Ca_i"}
    assert set(rest(ode, "mechanics").missing_variables) == set()


def test_torord_land_zeta_split(torord_land):
    m = torord_land.split("mechanics", inputs=["S", "W"], outputs=["Zs", "Zw", "Ta"])
    assert set(rest(m, "mechanics").missing_variables) == {"Zs", "Zw"}
    assert set(only(m, "mechanics").missing_variables) == {"S", "W"}
    # lambda reaches EP, as in the hand-made zeta split
    assert "lmbda" in names(rest(m, "mechanics").parameters)


def test_torord_land_zeta_split_as_simcardemsx_takes_it(torord_land):
    # simcardemsx's generate_ode_code, word for word
    ode = torord_land.split("mechanics", inputs=["S", "W"], outputs=["Zs", "Zw", "Ta"])
    mechanics_comp = ode.get_component("mechanics")
    mechanics_ode = mechanics_comp.to_ode()
    ep_ode = ode - mechanics_comp
    assert set(ep_ode.missing_variables) == {"Zs", "Zw"}
    assert set(mechanics_ode.missing_variables) == {"S", "W"}
    # Land's components that move keep their names, under the new component's
    assert ode["Zs"].components == ("mechanics.crossbridges",)
    assert ode["Ta"].components == ("mechanics.tension",)
    assert ode["S"].components == ("land.crossbridges",)


def test_torord_land_cai_split(torord_land):
    m = torord_land.split("mechanics", inputs=["cai"], outputs=["Ta", "J_TRPN"])
    assert set(rest(m, "mechanics").missing_variables) == {"J_TRPN"}
    assert set(only(m, "mechanics").missing_variables) == {"cai"}
    assert {"CaTRPN", "B", "S", "W", "Zs", "Zw"} <= names(m.get_component("mechanics").states)
    assert "Cd" in names(m.get_component("land").states)
    # trpnmax, ToR-ORd's, moves with J_TRPN, the only thing that reads it once the glue has
    # replaced Bcai: to mechanics itself, as it comes from outside the outputs' part
    assert torord_land["trpnmax"].components == ("ep.intracellular_ions", "land")
    assert m["trpnmax"].components == ("mechanics",)


def test_torord_land_bcai_has_no_troponin(torord_land):
    assert "trpnmax" not in {s.name for s in torord_land["Bcai"].expr.free_symbols}


def land_driven_by(tmp_path: Path, time_unit: str, ca: str) -> dict:
    """Land 2017 (in s) composed into a model in ``time_unit``, with ``Ca = ca``, and its
    generated Python code, generalized Rush-Larsen"""
    (tmp_path / f"{time_unit}.glue.ode").write_text(f'expressions("land")\nreplace Ca = {ca}\n')
    recipe = {
        "time_unit": time_unit,
        "glue": f"{time_unit}.glue.ode",
        "parts": [{"name": "land", "file": str(FILES / "land_2017.ode"), "time_unit": "s"}],
    }
    ode = gotranx.compose(recipe, base_dir=tmp_path)
    code = gotranx.cli.gotran2py.get_code(ode, scheme=[Scheme.generalized_rush_larsen])
    namespace: dict = {}
    exec(compile(code, f"<land in {time_unit}>", "exec"), namespace)
    return namespace


def grl(model: dict, dt: float, steps: int, every: int) -> np.ndarray:
    """The states every ``every`` steps, from 0 to ``steps``"""
    y = model["init_state_values"]()
    p = model["init_parameter_values"]()
    out = [y]
    for i in range(steps):
        y = model["generalized_rush_larsen"](y, i * dt, dt, p)
        if (i + 1) % every == 0:
            out.append(y)
    return np.array(out)


def test_land_in_s_and_composed_in_ms_agree(tmp_path):
    # The spec's check of time rescaling: one calcium transient (uM), written in each
    # model's own time unit, and the same step in each unit, 10 us, to 50 ms. Rush-Larsen,
    # as Land's B is too stiff for forward Euler at this step
    seconds = land_driven_by(tmp_path, "s", "0.1 + 1.0*exp(-((t - 0.02)/0.008)**2)")
    ms = land_driven_by(tmp_path, "ms", "0.1 + 1.0*exp(-((t - 20.0)/8.0)**2)")
    assert [seconds["state_index"](n) for n in ("CaTRPN", "B", "S", "W", "Cd")] == [
        ms["state_index"](n) for n in ("CaTRPN", "B", "S", "W", "Cd")
    ]
    in_s = grl(seconds, dt=1e-5, steps=5000, every=100)  # States every ms
    in_ms = grl(ms, dt=1e-2, steps=5000, every=100)
    # The transient moves the states, so agreeing is not agreeing at rest
    bound = seconds["state_index"]("CaTRPN")
    assert in_s[:, bound].max() > 10 * in_s[0, bound]
    # The same step in two units: round-off apart
    assert np.allclose(in_s, in_ms, rtol=1e-9, atol=1e-15)
