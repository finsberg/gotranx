import hashlib
import json
import tomllib
from pathlib import Path

import pytest
import sympy as sp

import gotranx
from gotranx import atoms
from gotranx.exceptions import CompositionError

here = Path(__file__).parent
FILES = here / "compose_files"
RECIPE = FILES / "toy_recipe.toml"
TOY_EP = FILES / "toy_ep.ode"
TOY = here / "odefiles" / "contraction_toy.ode"
BR = here / "odefiles" / "beeler_reuter_1977.ode"
NOBLE = here / "cellml_files" / "noble_1962.cellml"


def same_model(a: gotranx.ODE, b: gotranx.ODE, components: bool = True) -> None:
    """The same states and parameters, with equal values (and components), and the same
    assignments, each with ``simplify(a - b) == 0`` (and the same components)"""
    for kind in ("states", "parameters"):
        x = {s.name: s for s in getattr(a, kind)}
        y = {s.name: s for s in getattr(b, kind)}
        assert x.keys() == y.keys(), kind
        for name in x:
            # Numerically: a reparse can change a value's form (-1*84.624 to -1*1*84.624)
            assert float(x[name].value) == float(y[name].value), name
            if components:
                assert x[name].components == y[name].components, name
    x = {s.name: s for s in a.intermediates + a.state_derivatives}
    y = {s.name: s for s in b.intermediates + b.state_derivatives}
    assert x.keys() == y.keys()
    for name in x:
        assert type(x[name]) is type(y[name]), name
        if components:
            assert x[name].components == y[name].components, name
        assert sp.simplify(x[name].expr - y[name].expr) == 0, name


def toy_recipe(**extra) -> dict:
    """toy_recipe.toml as a dict, with absolute paths and without its glue"""
    return {
        "time_unit": "ms",
        "parts": [
            {"name": "ep", "file": str(TOY_EP), "time_unit": "ms"},
            {
                "name": "toy",
                "file": str(TOY),
                "time_unit": "s",
                "rename": {"dLambda": "dLambda_toy"},
            },
        ],
        **extra,
    }


def compose_glue(tmp_path: Path, glue: str, **extra) -> gotranx.ODE:
    """Compose the toy's two parts with ``glue``, written to ``glue.ode``"""
    (tmp_path / "glue.ode").write_text(glue)
    return gotranx.compose(toy_recipe(glue="glue.ode", **extra), base_dir=tmp_path)


def messages(recipe: dict, base_dir: Path | None = None) -> list[str]:
    """The messages of the CompositionError that composing ``recipe`` raises"""
    with pytest.raises(CompositionError) as e:
        gotranx.compose(recipe, base_dir=base_dir)
    assert str(e.value) == "\n".join(e.value.messages)
    return e.value.messages


def glue_messages(tmp_path: Path, glue: str, **extra) -> list[str]:
    (tmp_path / "glue.ode").write_text(glue)
    return messages(toy_recipe(glue="glue.ode", **extra), base_dir=tmp_path)


def test_recomposition_equals_hand_written():
    same_model(gotranx.compose(RECIPE), gotranx.load_ode(FILES / "toy_coupled.ode"))


def test_dict_recipe_equals_path_recipe():
    recipe = tomllib.loads(RECIPE.read_text())
    same_model(gotranx.compose(recipe, base_dir=FILES), gotranx.compose(RECIPE))
    same_model(gotranx.compose(recipe, base_dir=str(FILES)), gotranx.compose(str(RECIPE)))


def test_dict_recipe_resolves_against_the_current_directory(monkeypatch):
    monkeypatch.chdir(FILES)
    same_model(gotranx.compose(tomllib.loads(RECIPE.read_text())), gotranx.compose(RECIPE))


def test_each_part_is_one_component():
    ode = gotranx.compose(RECIPE)
    assert {c.name for c in ode.components} == {"ep", "toy"}
    assert gotranx.load_ode(TOY_EP).num_components == 2  # "calcium" and "gate"


def test_composed_model_has_no_missing_variables():
    assert gotranx.compose(RECIPE).missing_variables == {}


def test_identity_composition_equals_load_ode():
    # Review Focus 2: one part, no glue, the same time unit
    ode = gotranx.compose(
        {"time_unit": "ms", "parts": [{"name": "ep", "file": str(TOY_EP), "time_unit": "ms"}]}
    )
    assert {c.name for c in ode.components} == {"ep"}
    expected = gotranx.load_ode(TOY_EP)
    same_model(ode, expected, components=False)
    for atom in (*ode.states, *ode.parameters, *ode.intermediates, *ode.state_derivatives):
        assert atom.components == ("ep",), atom.name


def test_identity_composition_removes_singularities_once():
    # Review Focus 2, on a model with removable singularities. Their Taylor coefficients
    # depend on the expression's form at round-off, which a reparse changes, so the exact
    # comparison is made without the guards, and the guarded ones are compared by name.
    recipe = {"time_unit": "ms", "parts": [{"name": "br", "file": str(BR), "time_unit": "ms"}]}
    plain = gotranx.compose(recipe, remove_singularities=False)
    same_model(plain, gotranx.load_ode(BR, remove_singularities=False), components=False)

    def piecewise(ode):
        return {
            a.name for a in ode.intermediates + ode.state_derivatives if a.expr.has(sp.Piecewise)
        }

    ode = gotranx.compose(recipe)
    guarded = piecewise(gotranx.load_ode(BR)) - piecewise(
        gotranx.load_ode(BR, remove_singularities=False)
    )
    assert piecewise(ode) - piecewise(plain) == guarded != set()
    # Removal runs once, at the end, on the unguarded model
    assert ode == plain.remove_singularities()


def test_cellml_part():
    ode = gotranx.compose(
        {"time_unit": "ms", "parts": [{"name": "noble", "file": str(NOBLE), "time_unit": "ms"}]}
    )
    assert {c.name for c in ode.components} == {"noble"}
    imported = gotranx.myokit.cellml_to_gotran(NOBLE)
    assert {s.name for s in ode.states} == {s.name for s in imported.states}
    assert {p.name for p in ode.parameters} == {p.name for p in imported.parameters}


@pytest.mark.parametrize("unit", ["mV", "furlong", "banana"])
def test_bad_time_unit_names_the_part(unit):
    # Review Focus 3: not a time ("mV", "furlong"), or not a unit at all ("banana")
    recipe = toy_recipe()
    recipe["parts"][1]["time_unit"] = unit
    assert messages(recipe) == [f"part toy: time_unit {unit} is not a unit of time"]
    assert messages(toy_recipe(time_unit=unit)) == [f"time_unit {unit} is not a unit of time"]


@pytest.mark.parametrize(
    "value, message",
    [
        ("toy.cai", "cai is not a name of toy (it is ep's)"),  # Review Focus 4
        ("cai", "cai is not a name of toy (it is ep's)"),
        ("ep.SL0", "SL0 is not a name of ep (it is toy's)"),
        ("ep.nothing", "nothing is not a name of ep"),
        ("epp.cai", "there is no part epp (in epp.cai)"),
    ],
)
def test_qualified_name_of_the_wrong_part(tmp_path, value, message):
    glue = f'expressions("toy")\nreplace Ca = 1000.0*{value}\n'
    assert glue_messages(tmp_path, glue) == [f"glue.ode:2: {message}"]


def test_composed_model_saves_and_loads(tmp_path):
    # Review Focus 5: the header holds the recipe text, with quotes, `=`, `#` and newlines
    ode = gotranx.compose(RECIPE)
    ode.save(tmp_path / "m.ode")
    text = (tmp_path / "m.ode").read_text()
    assert '# time_unit = "ms"' in text and '# rename = { dLambda = "dLambda_toy" }' in text
    loaded = gotranx.load_ode(tmp_path / "m.ode", remove_singularities=False)
    same_model(loaded, ode)
    assert 'time_unit = "ms"' in loaded.text


def test_composing_twice_saves_identical_text(tmp_path):
    # No timestamp: simcardemsx caches generated code by the model file's sha256
    gotranx.compose(RECIPE).save(tmp_path / "a.ode")
    gotranx.compose(RECIPE).save(tmp_path / "b.ode")
    assert (tmp_path / "a.ode").read_bytes() == (tmp_path / "b.ode").read_bytes()


def test_header_records_sha256():
    ode = gotranx.compose(RECIPE)
    assert f"Composed by gotranx {gotranx.__version__} from toy_recipe.toml" in ode.text
    for name, path in [
        ("toy_recipe.toml", RECIPE),
        ("toy_glue.ode", FILES / "toy_glue.ode"),
        ("toy_ep.ode", TOY_EP),
        ("../odefiles/contraction_toy.ode", TOY),
    ]:
        assert f"sha256 {name}: {hashlib.sha256(path.read_bytes()).hexdigest()}" in ode.text
    for line in RECIPE.read_text().splitlines():
        if line and not line.startswith("#"):
            assert line.split("#")[0].strip() in ode.text


def test_header_of_a_dict_recipe():
    recipe = toy_recipe()
    ode = gotranx.compose(recipe)
    assert f"Composed by gotranx {gotranx.__version__} from a dict" in ode.text
    for line in json.dumps(recipe, sort_keys=True, indent=1).splitlines():
        assert line.strip() in ode.text
    assert f"sha256 {TOY_EP}: {hashlib.sha256(TOY_EP.read_bytes()).hexdigest()}" in ode.text


def test_cross_component_parameter_under_both(tmp_path):
    ode = compose_glue(tmp_path, 'expressions("toy")\nJ = ep.k_up*dCaTRPN_dt\n')
    assert ode["k_up"].components == ("ep", "toy")
    assert ode["c_rel"].components == ("ep",)
    assert ode["J"].components == ("toy",)
    ode.save(tmp_path / "m.ode")
    assert gotranx.load_ode(tmp_path / "m.ode")["k_up"].components == ("ep", "toy")


# One test per row of the spec's rules table


def test_new_name_already_in_part(tmp_path):
    assert glue_messages(tmp_path, 'parameters("toy",\nSL0 = 2.0\n)\n') == [
        "glue.ode:2: SL0 is already a parameter of toy: use it, or write `replace SL0 = ...`"
    ]
    assert glue_messages(tmp_path, 'expressions("toy")\nLambda = 2.0\n') == [
        "glue.ode:2: Lambda is already an intermediate of toy: use it, "
        "or write `replace Lambda = ...`"
    ]
    assert glue_messages(tmp_path, 'expressions("toy")\nB = 2.0\n') == [
        "glue.ode:2: B is already a state of toy: use it"
    ]
    assert glue_messages(tmp_path, 'expressions("toy")\ndB_dt = 2.0\n') == [
        "glue.ode:2: dB_dt is already a state derivative of toy: use it, or write `dB_dt += ...`"
    ]


def test_new_name_twice_in_the_glue(tmp_path):
    # Not "use it, or write `replace J = ...`", which would only give the next error
    assert glue_messages(tmp_path, 'expressions("toy")\nJ = 1.0\n\nJ = 2.0\n') == [
        "glue.ode:4: J is already defined in the glue, at line 2"
    ]


def test_new_name_in_another_part(tmp_path):
    assert glue_messages(tmp_path, 'expressions("toy")\nrelease = 2.0\n') == [
        "glue.ode:2: names must be unique: release is defined in ep; "
        "rename it in the recipe or choose another"
    ]
    glue = 'parameters("toy", x = 1.0)\nparameters("ep", x = 2.0)\n'
    assert glue_messages(tmp_path, glue) == [
        "glue.ode:2: names must be unique: x is defined in toy; "
        "rename it in the recipe or choose another"
    ]


def test_new_names(tmp_path):
    ode = compose_glue(tmp_path, 'parameters("toy", x = 2.0)\nexpressions("toy")\ny = x*B\n')
    assert isinstance(ode["x"], atoms.Parameter) and ode["x"].value == 2.0
    assert isinstance(ode["y"], atoms.Intermediate)
    assert ode["y"].expr == ode["x"].symbol * ode["B"].symbol
    assert ode["x"].components == ode["y"].components == ("toy",)


def test_replace_parameter_declaration(tmp_path):
    glue = (
        'parameters("toy",\n'
        'replace k_trpn = ScalarParam(200.0, unit="1/ms", description="faster")\n)\n'
    )
    ode = compose_glue(tmp_path, glue)
    k = ode["k_trpn"]
    assert isinstance(k, atoms.Parameter) and k.components == ("toy",)
    assert (k.value, k.unit_str, k.description) == (200.0, "1/ms", "faster")


def test_replace_parameter_with_expression_makes_an_input(tmp_path):
    ode = compose_glue(tmp_path, 'expressions("toy")\nreplace Ca = 1000.0*ep.cai\n')
    assert isinstance(ode["Ca"], atoms.Intermediate)
    assert ode["Ca"].components == ("toy",)
    assert "Ca" not in {p.name for p in ode.parameters}
    assert ode["Ca"].expr == 1000.0 * ode["cai"].symbol
    assert ode["Ca"].unit_str == "uM"  # The parameter's unit, as the glue gives none


def test_replace_intermediate(tmp_path):
    ode = compose_glue(tmp_path, 'expressions("toy")\nreplace Lambda = 2*SL/SL0\n')
    s = ode.symbols
    assert sp.simplify(ode["Lambda"].expr - 2 * s["SL"] / s["SL0"]) == 0
    assert ode["Lambda"].components == ("toy",)


def test_replace_state_derivative(tmp_path):
    ode = compose_glue(tmp_path, 'expressions("toy")\nreplace dZs_dt = -Zs\n')
    assert isinstance(ode["dZs_dt"], atoms.StateDerivative)
    assert ode["dZs_dt"].expr == -ode["Zs"].symbol


def test_replace_unknown_name(tmp_path):
    assert glue_messages(tmp_path, 'expressions("toy")\nreplace Caa = 1.0\n') == [
        "glue.ode:2: Caa is not a name of toy"
    ]
    assert glue_messages(tmp_path, 'parameters("toy",\nreplace k_up = 1.0\n)\n') == [
        "glue.ode:2: k_up is not a name of toy (it is ep's)"
    ]


def test_replace_twice(tmp_path):
    glue = 'expressions("toy")\nreplace Ca = 1.0\nreplace Ca = 2.0\nx = 1.0\nreplace x = 2.0\n'
    assert glue_messages(tmp_path, glue) == [
        "glue.ode:3: Ca is already defined in the glue, at line 2",
        "glue.ode:5: x is already defined in the glue, at line 4",
    ]


def test_add_to_derivative(tmp_path):
    plain = gotranx.compose(toy_recipe())
    ode = compose_glue(tmp_path, 'expressions("toy")\ndB_dt += -0.5*B\ndB_dt += 0.25*B\n')
    B = ode["B"].symbol
    assert sp.simplify(ode["dB_dt"].expr - plain["dB_dt"].expr + 0.25 * B) == 0
    assert isinstance(ode["dB_dt"], atoms.StateDerivative)


def test_add_to_non_derivative(tmp_path):
    assert glue_messages(tmp_path, 'expressions("toy")\nLambda += 1.0\n') == [
        "glue.ode:2: Lambda is not the derivative of a state of toy"
    ]
    assert glue_messages(tmp_path, 'expressions("toy")\ndcai_dt += 1.0\n') == [
        "glue.ode:2: dcai_dt is not the derivative of a state of toy (it is ep's)"
    ]


def test_replace_state_is_an_error(tmp_path):
    assert glue_messages(tmp_path, 'expressions("toy")\nreplace B = 0.5\n') == [
        "glue.ode:2: B is a state of toy: a state cannot be replaced"
    ]
    assert glue_messages(tmp_path, 'parameters("toy",\nreplace B = 0.5\n)\n') == [
        "glue.ode:2: B is a state of toy: a state cannot be replaced"
    ]


def test_replace_in_the_wrong_block(tmp_path):
    assert glue_messages(tmp_path, 'parameters("toy",\nreplace Lambda = 0.5\n)\n') == [
        "glue.ode:2: Lambda is an intermediate of toy: replace it in an expressions block"
    ]


def test_all_errors_reported_together(tmp_path):
    glue = (
        'expressions("toy")\nreplace Caa = 1.0\nrelease = 2.0\n'
        'expressions("ep")\ndn_dt += toy.cai\n'
    )
    with pytest.raises(CompositionError) as e:
        compose_glue(tmp_path, glue)
    assert e.value.messages == [
        "glue.ode:2: Caa is not a name of toy",
        "glue.ode:3: names must be unique: release is defined in ep; "
        "rename it in the recipe or choose another",
        "glue.ode:5: cai is not a name of toy (it is ep's)",
    ]
    assert str(e.value).splitlines() == e.value.messages


def test_missing_names_refused_unless_allowed(tmp_path):
    glue = 'expressions("toy")\nreplace Ca = 1000.0*cai_ext\nJ = x_ext + cai_ext\n'
    assert glue_messages(tmp_path, glue) == [
        "used but defined nowhere: cai_ext (glue.ode:2), x_ext (glue.ode:3); "
        "define them, or set allow_missing = true"
    ]
    ode = compose_glue(tmp_path, glue, allow_missing=True)
    assert ode.missing_variables == {"cai_ext": 0, "x_ext": 1}
    assert {c.name for c in ode.components} == {"ep", "toy"}
    assert isinstance(ode["Ca"], atoms.Intermediate)


def test_missing_names_with_a_part_named_missing(tmp_path):
    # The missing names are parameters of a component of their own while the model is
    # built, whose name must not be a part's
    recipe = toy_recipe(glue="glue.ode", allow_missing=True)
    recipe["parts"][1]["name"] = "missing"
    (tmp_path / "glue.ode").write_text('expressions("missing")\nreplace Ca = 1000.0*cai_ext\n')
    ode = gotranx.compose(recipe, base_dir=tmp_path)
    assert ode.missing_variables == {"cai_ext": 0}
    assert {c.name for c in ode.components} == {"ep", "missing"}
    assert len(ode.get_component("missing").states) == 3


def test_glue_parse_errors(tmp_path):
    assert glue_messages(tmp_path, 'expressions("toy")\nfoo x = 1\n') == [
        "glue.ode:2: unknown statement foo"
    ]
    assert glue_messages(tmp_path, '\nstates("toy", y = 1.0)\n') == [
        "glue.ode:2: glue files cannot declare states"
    ]
    assert glue_messages(tmp_path, 'expressions("toy")\nx = (1\n') == [
        "glue.ode:2: syntax error at column 7"
    ]


@pytest.mark.parametrize("value", ["ep.cai", "2*cai"])
def test_glue_parameter_values_must_be_numbers(tmp_path, value):
    assert glue_messages(tmp_path, f'parameters("toy",\nx = 1.0,\ny = {value}\n)\n') == [
        "glue.ode:3: parameter values must be numbers"
    ]


def test_glue_block_of_no_part(tmp_path):
    glue = 'expressions("tyo")\nx = 1.0\ny = 2.0\nexpressions("toy")\nz = 1.0\n'
    assert glue_messages(tmp_path, glue) == ["glue.ode:2: there is no part tyo"]


def test_missing_files(tmp_path):
    recipe = toy_recipe(glue="nowhere.ode")
    recipe["parts"][0]["file"] = "nothing.ode"
    assert messages(recipe, base_dir=tmp_path) == [
        "part ep: file nothing.ode not found",
        "glue file nowhere.ode not found",
    ]
    with pytest.raises(CompositionError, match="recipe file .*nothing.toml not found"):
        gotranx.compose(tmp_path / "nothing.toml")


def test_bad_part_file(tmp_path):
    (tmp_path / "bad.ode").write_text('expressions("a")\nx = y\n')
    recipe = {"time_unit": "ms", "parts": [{"name": "a", "file": "bad.ode", "time_unit": "ms"}]}
    assert messages(recipe, base_dir=tmp_path) == [
        "part a: bad.ode: Symbol 'y' not found in line 2"
    ]
    (tmp_path / "bad.ode").write_text('expressions("a")\nx = (1\n')
    assert messages(recipe, base_dir=tmp_path) == ["part a: bad.ode:2: syntax error at column 7"]


def test_bad_recipe_toml(tmp_path):
    (tmp_path / "r.toml").write_text("time_unit = \n")
    with pytest.raises(CompositionError, match="^r.toml: "):
        gotranx.compose(tmp_path / "r.toml")


def test_unknown_recipe_key_named():
    assert messages(toy_recipe(colour="red")) == [
        "unknown recipe key colour: the keys are time_unit, parts, glue and allow_missing"
    ]
    recipe = toy_recipe()
    recipe["parts"][1]["prefix"] = "t_"
    assert messages(recipe) == [
        "part toy: unknown key prefix: a part's keys are name, file, time_unit and rename"
    ]


def test_missing_recipe_keys():
    assert messages({"parts": toy_recipe()["parts"]}) == ["the recipe has no time_unit"]
    assert messages({"time_unit": "ms"}) == ["the recipe has no parts"]
    recipe = toy_recipe()
    del recipe["parts"][0]["file"]
    del recipe["parts"][1]["name"]
    assert messages(recipe) == ["part ep has no file", "part 2 has no name"]


@pytest.mark.parametrize(
    "recipe, message",
    [
        ({"time_unit": 1, "parts": toy_recipe()["parts"]}, "time_unit must be a string"),
        (
            {"time_unit": "ms", "parts": {"name": "ep"}},
            "parts must be a list of tables, one per part",
        ),
        ({"time_unit": "ms", "parts": ["ep"]}, "part 1 must be a table"),
        (
            {"time_unit": "ms", "parts": [{"name": "", "file": "a.ode", "time_unit": "ms"}]},
            "part 1: name must be a non-empty string",
        ),
        (
            {"time_unit": "ms", "parts": [{"name": "a", "file": 1, "time_unit": "ms"}]},
            "part a: file must be a string",
        ),
    ],
)
def test_recipe_structure(recipe, message):
    assert messages(recipe) == [message]


def test_recipe_value_types():
    recipe = toy_recipe(allow_missing="yes", glue=3)
    recipe["parts"][1]["rename"] = ["dLambda"]
    assert messages(recipe) == [
        "glue must be a string",
        "allow_missing must be true or false",
        "part toy: rename must be a table of names",
    ]


def test_duplicate_part_names():
    recipe = toy_recipe()
    recipe["parts"][1]["name"] = "ep"
    assert messages(recipe) == ["part names must be unique: 2 parts are named ep"]


def test_duplicate_names_across_parts():
    recipe = toy_recipe()
    recipe["parts"][1]["rename"] = {"dLambda": "dLambda_toy", "B": "n", "ku": "k_up"}
    assert messages(recipe) == [
        "names must be unique: k_up is defined in ep and in toy; rename it in the recipe",
        "names must be unique: n is defined in ep and in toy; rename it in the recipe",
    ]


def test_rename_unknown_name():
    recipe = toy_recipe()
    recipe["parts"][1]["rename"] = {
        "dLambda": "dLambda_toy",
        "nothing": "x",
        "B": "Zs",
        "Ta": "T a",
    }
    assert messages(recipe) == [
        "part toy: rename nothing: nothing is not a name of toy",
        "part toy: rename B: Zs is already a name of toy",
        "part toy: rename Ta: T a is not a valid name",
    ]


def test_rename_swap():
    recipe = toy_recipe()
    recipe["parts"][1]["rename"] = {"dLambda": "dLambda_toy", "B": "Zs", "Zs": "B"}
    ode = gotranx.compose(recipe)
    plain = gotranx.compose(toy_recipe())
    B, Zs = ode["B"].symbol, ode["Zs"].symbol
    swapped = plain["dB_dt"].expr.subs({B: Zs, Zs: B}, simultaneous=True)
    assert sp.simplify(ode["dZs_dt"].expr - swapped) == 0


def test_toy_recipe_needs_its_rename():
    # The reason for toy_recipe.toml's rename (ruling R1)
    recipe = tomllib.loads(RECIPE.read_text())
    del recipe["parts"][1]["rename"]
    assert messages(recipe, base_dir=FILES) == [
        "toy_glue.ode:6: dLambda is already an intermediate of toy: use it, "
        "or write `replace dLambda = ...`"
    ]


def test_new_derivative_without_a_state(tmp_path):
    # A glue file cannot declare states, so it cannot define a new state's derivative
    assert glue_messages(tmp_path, 'expressions("toy")\ndTension_dt = 1.0\n') == [
        "glue.ode:2: dTension_dt names the derivative of Tension, which is not a state of toy: "
        "choose another name"
    ]


@pytest.mark.parametrize(
    "glue, message",
    [
        (
            'expressions("toy")\nreplace Ca = 2*Ca\n',
            "glue.ode:2: Ca uses Ca: the equations are circular",
        ),
        (
            'expressions("ep")\nreplace release = toy.Ca*n\n'
            'expressions("toy")\nreplace Ca = 1000.0*ep.release\n',
            "glue.ode:2: release uses Ca uses release: the equations are circular",
        ),
        (
            # Through a state's derivative: J and dcai_dt each use the other
            'expressions("toy")\nJ = ep.dcai_dt\nexpressions("ep")\ndcai_dt += -toy.J\n',
            "glue.ode:2: J uses dcai_dt uses J: the equations are circular",
        ),
    ],
)
def test_circular_glue(tmp_path, glue, message):
    assert glue_messages(tmp_path, glue) == [message]


def test_circular_part(tmp_path):
    # gotranx loads a model whose equations are circular; composing it says so
    (tmp_path / "c.ode").write_text(
        'states("a", y=1.0)\nexpressions("a")\nx = 2*z\nz = x\ndy_dt = x\n'
    )
    recipe = {"time_unit": "ms", "parts": [{"name": "a", "file": "c.ode", "time_unit": "ms"}]}
    assert messages(recipe, base_dir=tmp_path) == ["x uses z uses x: the equations are circular"]
