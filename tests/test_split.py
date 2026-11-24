"""ODE.split: carve a component out of a model by what goes in and what comes out.

Most tests cut the toy composition (tests/compose_files/toy_recipe.toml) two ways, as
simcardemsx cuts ToR-ORd + Land: zeta-like (B in; Zs and Ta out) and Ca_i-like (cai in; Ta
and J out). Structure the toy lacks is built inline.
"""

from pathlib import Path

import pytest
from conftest import only, rest

import gotranx
from gotranx import operations
from gotranx.exceptions import SplitError
from gotranx.load import ode_from_string
from gotranx.ode import top_level

RECIPE = Path(__file__).parent / "compose_files" / "toy_recipe.toml"


def names(atoms) -> set[str]:
    return {a.name for a in atoms}


@pytest.fixture(scope="module")
def ode() -> gotranx.ODE:
    return gotranx.compose(RECIPE)


def split_messages(ode: gotranx.ODE, name: str, inputs, outputs) -> list[str]:
    """The messages of the SplitError that the split raises"""
    with pytest.raises(SplitError) as e:
        ode.split(name, inputs=inputs, outputs=outputs)
    assert str(e.value) == "\n".join(e.value.messages)
    return e.value.messages


def test_zeta_like_cut(ode):
    m = ode.split("mechanics", inputs=["B"], outputs=["Zs", "Ta"])
    # The toy's dB_dt does not read Zs, so the rest needs nothing back.
    assert set(rest(m, "mechanics").missing_variables) == set()
    assert set(only(m, "mechanics").missing_variables) == {"B"}
    mechanics = m.get_component("mechanics")
    assert names(mechanics.states) == {"Zs"}
    assert names(mechanics.intermediates) == {"Ta", "Lambda", "SL", "dLambda_toy", "dSL"}
    assert {"B", "CaTRPN"} <= names(m.get_component("toy").states)


def test_cai_like_cut(ode):
    m = ode.split("mechanics", inputs=["cai"], outputs=["Ta", "J"])
    assert set(rest(m, "mechanics").missing_variables) == {"J"}
    assert set(only(m, "mechanics").missing_variables) == {"cai"}
    assert "CaTRPN" in names(m.get_component("mechanics").states)  # pulled in from home
    assert names(m.get_component("mechanics").states) == {"CaTRPN", "B", "Zs"}
    assert "toy" not in {top_level(c.name) for c in m.components}  # every toy atom moved


def test_monitor_joins_the_component_whose_state_it_reads(ode):
    # bound_ca = CaTRPN is a monitor: no derivative depends on it.
    cai = ode.split("mechanics", inputs=["cai"], outputs=["Ta", "J"])
    assert cai["bound_ca"].components == ("mechanics.troponin",)
    zeta = ode.split("mechanics", inputs=["B"], outputs=["Zs", "Ta"])
    assert zeta["bound_ca"].components == ("toy.troponin",)


def test_moved_atoms_keep_their_sub_names(ode):
    # toy.crossbridges moves to mechanics.crossbridges, and what the glue put in toy to
    # mechanics
    m = ode.split("mechanics", inputs=["B"], outputs=["Zs", "Ta"])
    assert m["Zs"].components == m["dZs_dt"].components == ("mechanics.crossbridges",)
    assert m["dZs_dt"].state.components == ("mechanics.crossbridges",)
    assert m["Ta"].components == ("mechanics.tension",)
    assert m["lmbda"].components == ("mechanics",)
    # What stays keeps its name
    assert m["B"].components == ("toy.crossbridges",)
    assert {c.name for c in m.components} >= {"mechanics", "mechanics.crossbridges", "toy"}


def test_split_name_must_not_hold_existing_components(ode):
    # The composed model has no component "ep" of its own, but "ep" holds ep.calcium
    assert split_messages(ode, "ep", inputs=["B"], outputs=["Zs", "Ta"]) == [
        "ep is already a component of the model: choose another name"
    ]


BOTH_SIDES = """
parameters("a", p = 1.0, k = 2.0)
parameters("b", c = 3.0)
states("a", x = 0.0)
states("b", y = 0.0)
expressions("a")
q = 2*p
dx_dt = q - k*x
K = q*x
expressions("b")
dy_dt = q*x - c*y
"""


NESTED_BOTH_SIDES = """
parameters("a.kin", p = 1.0, k = 2.0)
parameters("b.x", c = 3.0, r = 4.0)
states("a.kin", x = 0.0)
states("b.x", y = 0.0)
states("b.z", z = 0.0)
expressions("a.kin")
q = 2*p
dx_dt = q - k*x
expressions("b.x")
s = 3*r
dy_dt = q*x + s - c*y
expressions("b.z")
dz_dt = s - z
"""


def test_shared_on_both_sides_gains_the_corresponding_sub_name():
    m = ode_from_string(NESTED_BOTH_SIDES, name="both").split("mech", inputs=["x"], outputs=["y"])
    # From the outputs' home, b: under the new component, in the same place
    assert m["s"].components == m["r"].components == ("b.x", "mech.x")
    for x in ("y", "dy_dt", "c"):
        assert m[x].components == ("mech.x",), x
    # From outside the home: under the new component itself, not as a mech.kin it never had
    assert m["q"].components == m["p"].components == ("a.kin", "mech")
    # What stays keeps its name
    assert m["k"].components == ("a.kin",)
    assert m["dz_dt"].components == ("b.z",)
    assert set(only(m, "mech").missing_variables) == {"x"}
    assert set(rest(m, "mech").missing_variables) == set()


def test_parameter_only_intermediates_on_both_sides():
    ode = ode_from_string(BOTH_SIDES, name="both")
    m = ode.split("mech", inputs=["x"], outputs=["y"])
    # q is computed on both sides, and p, which q reads, is written under both.
    assert set(m["q"].components) == {"a", "mech"}
    assert set(m["p"].components) == {"a", "mech"}
    assert "q" in names(only(m, "mech").intermediates)
    assert "q" in names(rest(m, "mech").intermediates)
    # Nothing else is duplicated: each side keeps its own.
    for x in ("x", "dx_dt", "k"):
        assert m[x].components == ("a",), x
    for x in ("y", "dy_dt", "c"):
        assert m[x].components == ("mech",), x
    assert m["dy_dt"].state.components == ("mech",)
    assert set(only(m, "mech").missing_variables) == {"x"}
    assert set(rest(m, "mech").missing_variables) == set()


def test_monitor_reading_a_shared_intermediate_stays():
    # K is a monitor, like Land's Ka in the zeta cut: it reads q, which is computed on
    # both sides, and x, which is an input. Neither ties it to the new component.
    m = ode_from_string(BOTH_SIDES, name="both").split("mech", inputs=["x"], outputs=["y"])
    assert m["K"].components == ("a",)


UNNAMED = """
parameters(p = 1.0, k = 2.0, c = 3.0)
states(x = 0.0, y = 0.0)
q = 2*p
dx_dt = q - k*x
dy_dt = q*x - c*y
"""


def test_split_a_model_without_component_names():
    # Its unnamed expressions must be printed before any named block, or they are read
    # back as part of it.
    m = ode_from_string(UNNAMED, name="unnamed").split("mech", inputs=["x"], outputs=["y"])
    assert m["q"].components == ("", "mech")
    assert m["dx_dt"].components == ("",)
    assert m["dy_dt"].components == ("mech",)
    assert set(only(m, "mech").missing_variables) == {"x"}


def test_parameter_only_intermediate_used_by_one_side_moves(ode):
    m = ode.split("mechanics", inputs=["B"], outputs=["Zs", "Ta"])
    moved = {
        "Lambda": "mechanics.kinematics",
        "SL": "mechanics.inputs",
        "SL0": "mechanics.kinetics",
        "lmbda": "mechanics",
        "Tref": "mechanics.kinetics",
    }
    for x, component in moved.items():
        assert m[x].components == (component,), x
    for x in ("kb", "ku"):  # read by dB_dt alone, which stays
        assert m[x].components == ("toy.kinetics",), x


def test_outputs_need_not_be_used_by_the_rest(ode):
    m = ode.split("mechanics", inputs=["B"], outputs=["Zs", "Ta"])
    assert "Ta" in names(only(m, "mechanics").intermediates)
    assert "Ta" not in rest(m, "mechanics").missing_variables


def test_reaching_another_component(ode):
    assert split_messages(ode, "mechanics", inputs=[], outputs=["Ta", "J"]) == [
        "mechanics needs cai from component ep.calcium: add it to inputs"
    ]


REST_READS = """
parameters("toy", kb = 1.0, cs = 1.0)
states("toy", B = 0.0, Zs = 0.0)
expressions("toy")
dB_dt = kb*(1 - B) - Zs
dZs_dt = -cs*Zs
Ta = B*(Zs + 1)
"""


def test_rest_uses_unlisted_output():
    ode = ode_from_string(REST_READS, name="toy")
    # Ta pulls in Zs from home, and the rest's dB_dt reads it.
    assert split_messages(ode, "mechanics", inputs=["B"], outputs=["Ta"]) == [
        "the rest uses Zs from mechanics: add it to outputs"
    ]
    m = ode.split("mechanics", inputs=["B"], outputs=["Ta", "Zs"])
    assert set(rest(m, "mechanics").missing_variables) == {"Zs"}
    assert set(only(m, "mechanics").missing_variables) == {"B"}


def test_unused_input(ode):
    assert split_messages(ode, "mechanics", inputs=["B", "n"], outputs=["Zs", "Ta"]) == [
        "n is not needed by mechanics"
    ]


def test_input_the_component_computes(ode):
    assert split_messages(ode, "mechanics", inputs=["B", "Zs"], outputs=["Zs", "Ta"]) == [
        "Zs is an input of mechanics, but mechanics computes it: remove it from inputs"
    ]


def test_unknown_name_and_existing_component(ode):
    messages = split_messages(ode, "toy", inputs=["B", "nope"], outputs=["Zs", "Ta", "Tb"])
    assert messages == [
        "nope is not a name of the model",
        "Tb is not a name of the model",
        "toy is already a component of the model: choose another name",
    ]


def test_all_errors_reported_together(ode):
    # Ta and J reach cai in ep, and n is not needed.
    assert split_messages(ode, "mechanics", inputs=["n"], outputs=["Ta", "J"]) == [
        "mechanics needs cai from component ep.calcium: add it to inputs",
        "n is not needed by mechanics",
    ]


def test_split_does_not_change_the_original(ode):
    before = {a.name: a for a in operations.all_atoms(ode)}
    ode.split("mechanics", inputs=["cai"], outputs=["Ta", "J"])
    assert {a.name: a for a in operations.all_atoms(ode)} == before
    assert ode == gotranx.compose(RECIPE)
    assert {top_level(c.name) for c in ode.components} == {"toy", "ep"}


def test_function_and_method_agree(ode):
    a = ode.split("mechanics", inputs=["cai"], outputs=["Ta", "J"])
    b = operations.split(ode, "mechanics", inputs=["cai"], outputs=["Ta", "J"])
    assert a == b
    assert a.name == ode.name


ELSEWHERE = """
parameters("a", k = 1.0)
parameters("b", c = 1.0)
states("a", x = 0.0)
states("b", y = 1.0)
expressions("a")
dx_dt = -k*x + y
T = 2*x
expressions("b")
dy_dt = -c*y
T_view = T*y
"""


def test_monitor_outside_home_stays_and_reads_the_output():
    # T_view is a monitor of b, which no output belongs to: it stays in b and reads the
    # output T as a crossing (ruling T5-b).
    m = ode_from_string(ELSEWHERE, name="elsewhere").split("mech", inputs=["y"], outputs=["T"])
    assert m["T_view"].components == ("b",)
    assert set(rest(m, "mech").missing_variables) == {"T"}
    assert set(only(m, "mech").missing_variables) == {"y"}


def test_unknown_input_does_not_hide_the_other_errors():
    ode = ode_from_string(ELSEWHERE, name="elsewhere")
    # A misspelled input: the closure still runs, so the genuine need for y is reported,
    # and yy is not also reported as not needed.
    assert split_messages(ode, "mech", inputs=["yy"], outputs=["T"]) == [
        "yy is not a name of the model",
        "mech needs y from component b: add it to inputs",
    ]


def test_inputs_and_outputs_must_not_be_strings(ode):
    with pytest.raises(TypeError, match="inputs must be a sequence of names, not a string"):
        ode.split("mechanics", inputs="B", outputs=["Zs", "Ta"])
    with pytest.raises(TypeError, match="outputs must be a sequence of names, not a string"):
        ode.split("mechanics", inputs=["B"], outputs="Ta")


def test_no_outputs(ode):
    assert split_messages(ode, "mechanics", inputs=["B"], outputs=[]) == [
        "mechanics needs at least one output"
    ]


def test_dotted_split_name_is_refused(ode):
    assert split_messages(ode, "toy.mech", ["B"], ["Zs", "Ta"]) == [
        "toy.mech: a split name cannot contain a dot"
    ]
