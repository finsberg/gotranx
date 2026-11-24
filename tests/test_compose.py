import hashlib
import inspect
import json
import tomllib
from pathlib import Path

import pytest
import sympy as sp

import gotranx
from gotranx import atoms, composition
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
            assert x[name].value == y[name].value, name
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_header(ode: gotranx.ODE, path: Path) -> list[str]:
    """The lines of the comment block ``ode.save(path)`` starts with"""
    ode.save(path)
    lines = path.read_text().splitlines()
    return lines[: lines.index("")]


TOY_HEADER = [
    f"# Composed by gotranx {gotranx.__version__} from toy_recipe.toml",
    *[f"# {line}" if line else "#" for line in RECIPE.read_text().splitlines()],
    f"# sha256 toy_recipe.toml: {sha256(RECIPE)}",
    f"# sha256 toy_glue.ode: {sha256(FILES / 'toy_glue.ode')}",
    f"# sha256 toy_ep.ode: {sha256(TOY_EP)}",
    f"# sha256 ../odefiles/contraction_toy.ode: {sha256(TOY)}",
]


def test_composed_model_saves_and_loads(tmp_path):
    # Review Focus 5: the header holds the recipe text, with quotes, `=`, `#` and newlines
    ode = gotranx.compose(RECIPE)
    assert saved_header(ode, tmp_path / "m.ode") == TOY_HEADER
    loaded = gotranx.load_ode(tmp_path / "m.ode", remove_singularities=False)
    same_model(loaded, ode)
    assert 'rename = { dLambda = "dLambda_toy" }  # the glue adds' in loaded.text


def test_saved_header_keeps_its_lines(tmp_path):
    # Each line as it is: the recipe can be read back from it, and no sha256 is split
    # from its file, though both are longer than the 80 columns other comments wrap at
    header = saved_header(gotranx.compose(RECIPE), tmp_path / "m.ode")
    assert header == TOY_HEADER
    assert all(len(line) > 80 for line in header if line.startswith("# sha256"))
    # The recipe, read back from the header: every line after the first, up to the hashes
    end = next(i for i, line in enumerate(header) if line.startswith("# sha256"))
    assert "".join(line[2:] + "\n" for line in header[1:end]) == RECIPE.read_text()


def test_saved_header_of_a_dict_recipe(tmp_path):
    recipe = toy_recipe()
    header = saved_header(gotranx.compose(recipe), tmp_path / "m.ode")
    assert header == [
        f"# Composed by gotranx {gotranx.__version__} from a dict",
        *[f"# {line}" for line in json.dumps(recipe, sort_keys=True, indent=1).splitlines()],
        f"# sha256 {TOY_EP}: {sha256(TOY_EP)}",
        f"# sha256 {TOY}: {sha256(TOY)}",
    ]


def test_operations_on_a_composed_model_keep_its_header(tmp_path):
    ode = gotranx.compose(RECIPE).rename({"J": "J_TRPN"}).rescale_time(1.0)
    assert saved_header(ode, tmp_path / "m.ode") == TOY_HEADER


def test_compose_is_the_function_and_composition_the_module():
    assert inspect.ismodule(gotranx.composition)
    assert gotranx.compose is composition.compose


def test_composing_twice_saves_identical_text(tmp_path):
    # No timestamp: simcardemsx caches generated code by the model file's sha256
    gotranx.compose(RECIPE).save(tmp_path / "a.ode")
    gotranx.compose(RECIPE).save(tmp_path / "b.ode")
    assert (tmp_path / "a.ode").read_bytes() == (tmp_path / "b.ode").read_bytes()


def test_header_records_sha256():
    ode = gotranx.compose(RECIPE)
    for line in TOY_HEADER:
        assert line[2:] in ode.text


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


@pytest.mark.parametrize(
    "line, expected",
    [
        ("replace Ca = 0.2", (0.2, "uM", "cytosolic calcium")),
        ("replace Ca = 0.2 # mM", (0.2, "uM", "cytosolic calcium")),  # Not a unit
        ('replace Ca = ScalarParam(0.2, unit="mM")', (0.2, "mM", "cytosolic calcium")),
        ('replace Ca = ScalarParam(0.2, description="free")', (0.2, "uM", "free")),
    ],
)
def test_replace_parameter_keeps_its_unit_and_description(tmp_path, line, expected):
    # Unless the glue line gives them, as a replace in an expressions block keeps the unit
    ode = compose_glue(tmp_path, f'parameters("toy",\n{line}\n)\n')
    ca = ode["Ca"]
    assert isinstance(ca, atoms.Parameter) and ca.components == ("toy",)
    assert (ca.value, ca.unit_str, ca.description) == expected
    assert ca.unit == gotranx.units.ureg(expected[1]).units


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
        "unknown recipe key colour: the keys are time_unit, parts, glue, allow_missing "
        "and components"
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
            {"time_unit": "ms", "parts": [{"name": 3, "file": "a.ode", "time_unit": "ms"}]},
            "part 1: name must be a string",
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


def test_rename_onto_a_name_whose_rename_is_refused():
    # Zs's rename is refused, so Zs stays and B cannot take its name (this raised a raw
    # DuplicateSymbolError); the part's other renames are still made
    recipe = toy_recipe()
    recipe["parts"][1]["rename"] = {"dLambda": "dLambda_toy", "B": "Zs", "Zs": "t"}
    assert messages(recipe) == [
        "part toy: rename B: Zs is already a name of toy",
        "part toy: rename Zs: t is the name of time: choose another name",
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


def test_recipe_errors_do_not_hide_glue_errors(tmp_path):
    assert glue_messages(tmp_path, 'expressions("toy")\nreplace Caa = 1.0\n', colour="red") == [
        "unknown recipe key colour: the keys are time_unit, parts, glue, allow_missing "
        "and components",
        "glue.ode:2: Caa is not a name of toy",
    ]


def test_errors_that_stop_the_composition(tmp_path):
    # A part with an unusable time unit cannot be composed, so the glue is not checked
    (tmp_path / "glue.ode").write_text('expressions("toy")\nreplace Caa = 1.0\n')
    recipe = toy_recipe(glue="glue.ode", colour="red")
    recipe["parts"][1]["time_unit"] = "mV"
    assert messages(recipe, base_dir=tmp_path) == [
        "unknown recipe key colour: the keys are time_unit, parts, glue, allow_missing "
        "and components",
        "part toy: time_unit mV is not a unit of time",
    ]


def test_key_messages_follow_the_key_lists(monkeypatch):
    # So that a new key is listed without editing the messages
    monkeypatch.setattr(composition, "RECIPE_KEYS", (*composition.RECIPE_KEYS, "extra"))
    monkeypatch.setattr(composition, "PART_KEYS", (*composition.PART_KEYS, "prefix"))
    recipe = toy_recipe(colour="red", components={})
    recipe["parts"][1].update(kind="x", prefix="t_")
    assert messages(recipe) == [
        "unknown recipe key colour: the keys are time_unit, parts, glue, allow_missing, "
        "components and extra",
        "part toy: unknown key kind: a part's keys are name, file, time_unit, rename and prefix",
    ]


@pytest.mark.parametrize(
    "rename, message",
    [
        (
            {"dB_dt": "x"},
            "part toy: rename dB_dt: dB_dt is the derivative of B: rename B, "
            "and its derivative is renamed with it",
        ),
        (
            {"Lambda": "dTension_dt"},
            "part toy: rename Lambda: dTension_dt names the derivative of Tension, "
            "which is not a state of toy: choose another name",
        ),
        ({"Zs": "t"}, "part toy: rename Zs: t is the name of time: choose another name"),
        ({"Zs": "time"}, "part toy: rename Zs: time is the name of time: choose another name"),
    ],
)
def test_rename_of_derivatives_and_time(rename, message):
    recipe = toy_recipe()
    recipe["parts"][1]["rename"] = {"dLambda": "dLambda_toy", **rename}
    assert messages(recipe) == [message]


@pytest.mark.parametrize(
    "glue, line",
    [('expressions("toy")\nt = 2.0\n', "t"), ('parameters("toy",\ntime = 1.0\n)\n', "time")],
)
def test_new_glue_name_is_time(tmp_path, glue, line):
    assert glue_messages(tmp_path, glue) == [
        f"glue.ode:2: {line} is the name of time: choose another name"
    ]


@pytest.mark.parametrize("name", ["a.b", "my toy", "", "1x"])
def test_part_names_must_be_valid_names(name):
    # Glue writes part.name, so a part's name must be a name
    recipe = toy_recipe()
    recipe["parts"][1]["name"] = name
    assert messages(recipe) == [
        f"part 2: name {name!r} is not a valid name: use letters, digits and underscores, "
        "not starting with a digit"
    ]


@pytest.mark.parametrize(
    "line, unit",
    [
        ("replace Ca = 1000.0*ep.cai", "uM"),
        ("replace Ca = 1000.0*ep.cai # calcium", "uM"),  # A comment is not a unit
        ("replace Ca = 1000.0*ep.cai # mM", "mM"),  # The glue's unit wins
    ],
)
def test_input_keeps_the_parameter_unit(tmp_path, line, unit):
    ode = compose_glue(tmp_path, f'expressions("toy")\n{line}\n')
    assert ode["Ca"].unit_str == unit


def test_components_table_equals_split_after():
    plain = gotranx.compose(toy_recipe(glue=str(FILES / "toy_glue.ode")))
    recipe = toy_recipe(
        glue=str(FILES / "toy_glue.ode"),
        components={"mechanics": {"inputs": ["toy.B"], "outputs": ["toy.Zs", "toy.Ta"]}},
    )
    same_model(
        gotranx.compose(recipe), plain.split("mechanics", inputs=["B"], outputs=["Zs", "Ta"])
    )


def test_components_tables_split_before_the_guards():
    # split follows each assignment's parse tree, which is that of the unguarded model,
    # so the tables are applied first, and the guards to their result
    recipe = {"time_unit": "ms", "parts": [{"name": "br", "file": str(BR), "time_unit": "ms"}]}
    plain = gotranx.compose(recipe, remove_singularities=False)
    split = plain.split("m_gate", inputs=["V"], outputs=["m"])
    tables = {"m_gate": {"inputs": ["br.V"], "outputs": ["br.m"]}}
    # Exactly: the guards' Taylor coefficients differ at round-off if split after them.
    # The headers differ, as one recipe has the tables
    guarded = split.remove_singularities()
    assert guarded["alpha_m"].expr.has(sp.Piecewise)
    composed = gotranx.compose({**recipe, "components": tables})
    assert composed.components == guarded.components
    composed = gotranx.compose({**recipe, "components": tables}, remove_singularities=False)
    assert composed.components == split.components


def test_components_table_checks_qualifiers():
    recipe = toy_recipe(
        glue=str(FILES / "toy_glue.ode"),
        components={"mechanics": {"inputs": ["toy.B"], "outputs": ["ep.Zs"]}},
    )
    assert any("Zs is not a name of ep (it is toy's)" in m for m in messages(recipe))


@pytest.mark.parametrize(
    "table, expected",
    [
        (
            {"inputs": ["toy.B", "ep.n"], "outputs": ["toy.Zs", "toy.Ta"]},
            ["components.mechanics: n is not needed by mechanics"],
        ),
        (
            {"inputs": [], "outputs": ["toy.Zs", "toy.Ta"]},
            [
                "components.mechanics: mechanics needs cai from component ep: add it to inputs",
                "components.mechanics: the rest uses dCaTRPN_dt from mechanics: add it to outputs",
            ],
        ),
    ],
)
def test_components_table_split_errors_are_composition_errors(table, expected):
    # The names are the parts', so the split itself fails, with every message of its
    # SplitError
    recipe = toy_recipe(glue=str(FILES / "toy_glue.ode"), components={"mechanics": table})
    assert messages(recipe) == expected


def test_components_table_bare_names():
    # A bare name is not checked against a part: the split checks it
    def compose(inputs, outputs):
        tables = {"mechanics": {"inputs": inputs, "outputs": outputs}}
        return gotranx.compose(toy_recipe(glue=str(FILES / "toy_glue.ode"), components=tables))

    bare = compose(["B"], ["Zs", "Ta"])
    assert bare.components == compose(["toy.B"], ["toy.Zs", "toy.Ta"]).components
    assert {c.name for c in bare.components} == {"ep", "toy", "mechanics"}
    with pytest.raises(CompositionError) as e:
        compose(["B"], ["Zs", "nonsense"])
    assert e.value.messages == ["components.mechanics: nonsense is not a name of the model"]


@pytest.mark.parametrize(
    "components, expected",
    [
        (3, ["components must be a table of tables, one per component"]),
        ({"mechanics": 3}, ["components.mechanics must be a table with inputs and outputs"]),
        (
            {"mechanics": {"inputs": "B"}},
            [
                "components.mechanics: inputs must be a list of names",
                "components.mechanics has no outputs",
            ],
        ),
        (
            {"mechanics": {"inputs": ["B"], "outputs": ["Zs", 2], "colour": "red"}},
            [
                "components.mechanics: unknown key colour: a component's keys are inputs and "
                "outputs",
                "components.mechanics: outputs must be a list of names",
            ],
        ),
    ],
)
def test_components_table_structure(components, expected):
    assert messages(toy_recipe(components=components)) == expected


def test_input_files(tmp_path):
    assert composition.input_files(RECIPE) == [
        ("the recipe", RECIPE),
        ("the glue file", FILES / "toy_glue.ode"),
        ("the file of part ep", FILES / "toy_ep.ode"),
        ("the file of part toy", FILES / "../odefiles/contraction_toy.ode"),
    ]
    recipe = toy_recipe(glue="glue.ode")
    recipe["parts"][0]["file"] = "ep.ode"
    del recipe["parts"][1]["name"]
    assert composition.input_files(recipe, base_dir=tmp_path) == [
        ("the glue file", tmp_path / "glue.ode"),
        ("the file of part ep", tmp_path / "ep.ode"),
        ("the file of part 2", TOY),
    ]
    # Nothing is checked: a recipe that cannot be read is the only file
    (tmp_path / "r.toml").write_text("time_unit = \n")
    assert composition.input_files(tmp_path / "r.toml") == [("the recipe", tmp_path / "r.toml")]


def test_recipe_and_glue_are_read_as_utf8(tmp_path, monkeypatch):
    # TOML is UTF-8, whatever the locale's encoding: here Latin-1, for files read without
    # one
    read_text = Path.read_text

    def latin1_by_default(self, encoding=None, *args, **kwargs):
        return read_text(self, encoding or "latin-1", *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", latin1_by_default)
    glue = 'parameters("ep",\nx = ScalarParam(1.0, description="Ca²⁺ in µM")\n)\n'
    (tmp_path / "glue.ode").write_text(glue, encoding="utf-8")
    recipe = tmp_path / "r.toml"
    recipe.write_text(
        '# Ca²⁺ in µM\ntime_unit = "ms"\nglue = "glue.ode"\n'
        f'[[parts]]\nname = "ep"\nfile = "{TOY_EP.as_posix()}"\ntime_unit = "ms"\n',
        encoding="utf-8",
    )
    ode = gotranx.compose(recipe)
    assert ode["x"].description == "Ca²⁺ in µM"
    assert ode.comments[1].text == "# Ca²⁺ in µM"


def test_recipe_and_glue_must_be_utf8(tmp_path):
    (tmp_path / "glue.ode").write_bytes('# Ca in µM\nparameters("ep", x = 1.0)\n'.encode("latin-1"))
    recipe = toy_recipe(glue="glue.ode")
    assert messages(recipe, base_dir=tmp_path) == [
        "glue.ode: not UTF-8 (invalid start byte at position 8)"
    ]
    (tmp_path / "r.toml").write_bytes('# Ca in µM\ntime_unit = "ms"\n'.encode("latin-1"))
    with pytest.raises(CompositionError) as e:
        gotranx.compose(tmp_path / "r.toml")
    assert e.value.messages == ["r.toml: not UTF-8 (invalid start byte at position 8)"]
