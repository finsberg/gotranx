import pytest
import sympy as sp

from gotranx import exceptions, operations
from gotranx.load import ode_from_string

TEXT = """
states("c", y=1.0)
parameters("c", a=2.0, b=3.0)
expressions("c")
x = a*b
J = 2*dy_dt
dy_dt = -a*y
"""

ode = ode_from_string(TEXT, name="m")


def names(atoms):
    return {a.name for a in atoms}


def test_rename_parameter_everywhere():
    new = ode.rename({"a": "q"})
    assert "q" in names(new.parameters) and "a" not in names(new.parameters)
    assert new["x"].expr == new.symbols["q"] * new.symbols["b"]
    assert new["q"].value == 2.0


def test_rename_state_renames_its_derivative():
    new = ode.rename({"y": "z"})
    assert "dz_dt" in names(new.state_derivatives)
    assert "dy_dt" not in names(new.state_derivatives)
    assert new["J"].expr == 2 * new.symbols["dz_dt"]


def test_rename_unknown_name_raises():
    with pytest.raises(KeyError, match="nope"):
        ode.rename({"nope": "q"})


def test_rename_onto_existing_name_raises():
    with pytest.raises(exceptions.DuplicateSymbolError):
        ode.rename({"a": "b"})


def test_rename_keeps_components_and_the_original():
    new = ode.rename({"a": "q"})
    assert new["q"].components == ("c",)
    assert "a" in names(ode.parameters)


def test_rebuild_round_trips():
    assert operations.rebuild(operations.all_atoms(ode), name="m", comments=ode.comments) == ode


def test_rename_symbol_is_canonical():
    new = ode.rename({"a": "q"})
    assert isinstance(new["q"].symbol, sp.Symbol)
    assert new["q"].symbol == new.symbols["q"]


MULTI = """
states("a", y=1.0)
parameters("a", "b", p=1.0)
expressions("a")
dy_dt = -p*y
expressions("b")
w = 2*p
"""


def test_rebuild_multi_component_atom():
    multi = ode_from_string(MULTI, name="m")
    assert multi["p"].components == ("a", "b")

    first = operations.rebuild(operations.all_atoms(multi), name="m")
    assert names(first.get_component("a").parameters) == {"p"}
    assert names(first.get_component("b").parameters) == {"p"}
    assert first["p"].components == ("a", "b")

    second = operations.rebuild(operations.all_atoms(first), name="m")
    assert second == first
