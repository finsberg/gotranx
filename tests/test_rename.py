import pytest
import sympy as sp

from gotranx import atoms, exceptions, operations
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


NEGATIVE = """
parameters("a", g = -2.5, k = 3)
states("a", V = ScalarParam(-84.624, unit="mV"), x = 1.0)
expressions("a")
y = x - 3
p = x**-2
h = Conditional(Lt(x, -1), -x, x - 1)
dV_dt = -(V - g)/k
dx_dt = -x + y
"""


def test_rebuild_is_idempotent(tmp_path):
    # The parser reads -84.624 as -1 times 84.624, which used to print as -1*84.624, read
    # back as (-1*1)*84.624, and so on: every rebuild added a factor
    original = ode_from_string(NEGATIVE, name="m", remove_singularities=False)
    ode = original
    texts = []
    for _ in range(3):
        ode = operations.rebuild(operations.all_atoms(ode), name="m")
        ode.save(tmp_path / "m.ode")
        texts.append((tmp_path / "m.ode").read_text())
    assert texts[0] == texts[1] == texts[2]
    lines = texts[0].splitlines()
    assert 'V=ScalarParam(-84.624, unit="mV"),' in lines
    assert "g=-2.5," in lines
    assert "y = x - 3" in lines
    assert "p = x**(-2)" in lines
    assert "*1" not in texts[0] and "-1*" not in texts[0]
    # The atoms are as parsed, not only equal in value
    for atom in (*original.states, *original.parameters):
        assert ode[atom.name].value == atom.value
    for atom in (*original.intermediates, *original.state_derivatives):
        assert ode[atom.name].expr == atom.expr


def test_rebuild_keeps_the_comments_it_is_given(tmp_path):
    line = "sha256 a/long/path/to/the/model.ode: " + 64 * "f"
    comments = [atoms.Comment(line, verbatim=True), atoms.Comment("wrapped " * 15)]
    new = operations.rebuild(operations.all_atoms(ode), name="m", comments=comments)
    assert new.comments == tuple(comments)
    assert [c.verbatim for c in new.comments] == [True, False]
    new.save(tmp_path / "m.ode")
    text = (tmp_path / "m.ode").read_text().splitlines()
    assert text[0] == f"# {line}"  # Verbatim: one line, however long
    # Wrapped at 80 columns, as before (onto a line that the wrapper starts with "# # ")
    assert text[1] == "# " + " ".join(10 * ["wrapped"])
    assert text[2].endswith(" ".join(5 * ["wrapped"])) and text[3] == ""
