"""Composition on real models: TT06 + Land 2017 (one way), ToR-ORd + Land 2017 (two way).

The hand-made splits in simcardemsx name the same things differently: ``XS``/``XW`` there
are ``S``/``W`` here, ``Zetas``/``Zetaw`` are ``Zs``/``Zw``, and ``CaTrpn`` is ``CaTRPN``.
"""

from pathlib import Path

import pytest
from conftest import only, rest

import gotranx

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


def test_torord_land_cai_split(torord_land):
    m = torord_land.split("mechanics", inputs=["cai"], outputs=["Ta", "J_TRPN"])
    assert set(rest(m, "mechanics").missing_variables) == {"J_TRPN"}
    assert set(only(m, "mechanics").missing_variables) == {"cai"}
    assert {"CaTRPN", "B", "S", "W", "Zs", "Zw"} <= names(m.get_component("mechanics").states)
    assert "Cd" in names(m.get_component("land").states)


def test_torord_land_bcai_has_no_troponin(torord_land):
    assert "trpnmax" not in {s.name for s in torord_land["Bcai"].expr.free_symbols}
