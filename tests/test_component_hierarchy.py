"""A dot in a component name nests it: ``"ep.gate"`` is under ``"ep"``, and ``"a.b.c"`` under
``"a.b"`` and ``"a"``. ``ODE.get_component`` and ``ODE.__sub__`` take a component together with
everything nested under it. Each atom still lists the names of the components it is in, so
the nesting is a convention on the names only."""

import pytest
import sympy as sp
import typer

from gotranx import load_ode
from gotranx.cli.utils import select_components
from gotranx.load import ode_from_string
from gotranx.ode import in_subtree, top_level

NESTED = """
parameters("ep.a", p = 1.0)
parameters("ep.b", "other", k = 2.0)
states("ep.a", x = 1.0)
states("ep.b", y = 1.0)
states("other", z = 1.0)
expressions("ep.a")
dx_dt = -p*x + y
expressions("ep.b")
dy_dt = -k*y
expressions("other")
dz_dt = -k*z + x
"""

TWO_LEVELS = """
states("a", u = 1.0)
states("a.b", v = 1.0)
states("a.b.c", w = 1.0)
states("a.bc", s = 1.0)
expressions("a")
du_dt = -u
expressions("a.b")
dv_dt = -v + u
expressions("a.b.c")
dw_dt = -w + v
expressions("a.bc")
ds_dt = -s + w
"""

# A model without dots, whose blocks name several components: an atom in such a block is
# in each of them, as simcardemsx's hand-made split files have it
SHARED = """
parameters("a", "mech", p = 1.0)
states("a", x = 1.0)
states("mech", y = 1.0)
expressions("a", "mech")
q = 2*p
expressions("a")
dx_dt = q - x + y
expressions("mech")
dy_dt = q*x - y
"""


def names(atoms) -> set[str]:
    return {a.name for a in atoms}


def component_names(ode) -> set[str]:
    return {c.name for c in ode.components}


@pytest.mark.parametrize(
    "component, name, expected",
    [
        ("ep", "ep", True),
        ("ep.a", "ep", True),
        ("a.b.c", "a.b", True),
        ("a.b.c", "a", True),
        ("epa", "ep", False),  # A prefix of the text, not of the dotted name
        ("a.bc", "a.b", False),
        ("ep", "ep.a", False),
    ],
)
def test_in_subtree(component, name, expected):
    assert in_subtree(component, name) is expected


@pytest.mark.parametrize(
    "component, expected", [("ep", "ep"), ("ep.a", "ep"), ("a.b.c", "a"), ("", "")]
)
def test_top_level(component, expected):
    assert top_level(component) == expected


def test_get_component_is_the_union_of_the_sub_components():
    ode = ode_from_string(NESTED, name="nested")
    assert "ep" not in component_names(ode)
    ep = ode.get_component("ep")
    assert ep.name == "ep"
    assert names(ep.states) == {"x", "y"}
    assert names(ep.parameters) == {"p", "k"}
    assert names(ep.state_derivatives) == {"dx_dt", "dy_dt"}
    # The atoms are those of the model, with their own components
    assert ode["k"] in ep.parameters
    assert ode["k"].components == ("ep.b", "other")


def test_get_component_to_ode():
    ode = ode_from_string(NESTED, name="nested")
    ep = ode.get_component("ep").to_ode()
    assert ep.name == "ep"
    assert names(ep.states) == {"x", "y"}
    assert ep.missing_variables == {}
    other = ode.get_component("other").to_ode()
    assert other.missing_variables == {"x": 0}


def test_get_component_of_a_leaf_is_the_component_itself():
    ode = ode_from_string(NESTED, name="nested")
    a = ode.get_component("ep.a")
    assert any(a is c for c in ode.components)


def test_get_component_with_atoms_of_its_own_and_sub_components():
    ode = ode_from_string(TWO_LEVELS, name="two")
    assert names(ode.get_component("a").states) == {"u", "v", "w", "s"}
    assert names(ode.get_component("a.b").states) == {"v", "w"}
    assert names(ode.get_component("a.bc").states) == {"s"}
    assert names(ode.get_component("a.b.c").states) == {"w"}


@pytest.mark.parametrize("name", ["nope", "e", "ep.c", "ep."])
def test_get_component_unknown_raises_key_error(name):
    ode = ode_from_string(NESTED, name="nested")
    with pytest.raises(KeyError):
        ode.get_component(name)


def test_subtract_removes_the_sub_components():
    ode = ode_from_string(NESTED, name="nested")
    rest = ode - ode.get_component("ep")
    assert component_names(rest) == {"other"}
    assert names(rest.states) == {"z"}
    # k is also in "other", so it stays
    assert names(rest.parameters) == {"k"}
    assert rest.missing_variables == {"x": 0}
    assert rest.name == "nested - ep"


def test_subtract_at_two_levels():
    ode = ode_from_string(TWO_LEVELS, name="two")
    rest = ode - ode.get_component("a.b")
    assert component_names(rest) == {"a", "a.bc"}
    assert rest.missing_variables == {"w": 0}
    only = ode.get_component("a.b").to_ode()
    assert only.missing_variables == {"u": 0}


def test_subtract_matches_by_name():
    # A component of the same name, from another model, removes this model's
    ode = ode_from_string(NESTED, name="nested")
    renamed = ode.rename({"p": "p2"})
    assert renamed.get_component("ep") != ode.get_component("ep")
    assert component_names(renamed - ode.get_component("ep")) == {"other"}


def test_a_model_without_dots_is_unchanged():
    ode = ode_from_string(SHARED, name="shared")
    mech = ode.get_component("mech")
    assert any(mech is c for c in ode.components)
    rest = ode - mech
    assert list(rest.components) == [c for c in ode.components if c.name != "mech"]
    # q and p are in "a" too, so they stay
    assert names(rest.intermediates) == {"q"}
    assert names(rest.parameters) == {"p"}
    assert rest.missing_variables == {"y": 0}
    assert mech.to_ode().missing_variables == {"x": 0}


def test_save_and_load_keep_the_dotted_names(tmp_path):
    ode = ode_from_string(NESTED, name="nested")
    ode.save(tmp_path / "nested.ode")
    text = (tmp_path / "nested.ode").read_text()
    assert 'states("ep.a",' in text
    assert 'parameters("ep.b", "other",' in text
    loaded = load_ode(tmp_path / "nested.ode")
    assert component_names(loaded) == component_names(ode) == {"ep.a", "ep.b", "other"}
    assert loaded["k"].components == ("ep.b", "other")
    assert names(loaded.get_component("ep").states) == {"x", "y"}


def test_rescale_time_selects_the_sub_components():
    ode = ode_from_string(NESTED, name="nested")
    scaled = ode.rescale_time(0.5, components=["ep"])
    for name in ("dx_dt", "dy_dt"):
        assert sp.simplify(scaled[name].expr - 0.5 * ode[name].expr) == 0, name
    assert sp.simplify(scaled["dz_dt"].expr - ode["dz_dt"].expr) == 0
    with pytest.raises(ValueError, match="nested has no component e:"):
        ode.rescale_time(0.5, components=["e"])


def test_cli_selects_the_sub_components():
    ode = ode_from_string(NESTED, name="nested")
    kept = select_components(ode, keep="ep")
    assert names(kept.states) == {"x", "y"}
    removed = select_components(ode, remove=["ep"])
    assert names(removed.states) == {"z"}
    assert removed.name == "nested_without_ep"
    with pytest.raises(typer.BadParameter, match="no component 'e' in nested"):
        select_components(ode, keep="e")


@pytest.mark.parametrize("remove", [["ep", "ep.a"], ["ep.a", "ep"]])
def test_cli_remove_overlapping_names(remove):
    ode = ode_from_string(NESTED, name="nested")
    assert names(select_components(ode, remove=remove).states) == {"z"}
