import os
import re
import subprocess
import sys
from pathlib import Path

from gotranx.templates import cytozoo


def test_model_struct_has_one_parameters_field():
    code = cytozoo.model_struct("ORdmmLand", 177)
    assert "struct ORdmmLand{T <: AbstractVector} <: AbstractCardiacCellModel" in code
    assert "parameters::T" in code
    # ToRORd carries celltype and stim; a generated model must not invent them.
    assert "celltype" not in code
    assert "stim" not in code
    # Three constructors, matching ToRORd's shape.
    assert "ORdmmLand() = ORdmmLand(Float64)" in code
    assert "function ORdmmLand(::Type{ElT}) where {ElT <: Number}" in code
    assert "function ORdmmLand(::Type{VT}) where {VT <: AbstractVector}" in code


def test_index_lookups_raise_rather_than_return_minus_one():
    code = cytozoo.index_lookups("ORdmmLand", ["v", "cai"], ["GNa"])
    assert "state_index(::ORdmmLand, name::Symbol)" in code
    assert "parameter_index(::ORdmmLand, name::Symbol)" in code
    # A Dict lookup raises KeyError on an unknown name; -1 must appear nowhere.
    assert "-1" not in code


def test_name_tuples_tuple_arity_cases():
    # Empty list case: must produce () with no trailing comma
    code = cytozoo.name_tuples("ORd", [], [], [])
    assert "ORD_STATE_NAMES = ()" in code
    assert "ORD_PARAMETER_NAMES = ()" in code
    assert "ORD_MONITOR_NAMES = ()" in code

    # One-element list case: must produce trailing comma to form a tuple, not a bare symbol
    code = cytozoo.name_tuples("ORd", ["v"], ["GNa"], ["Calcium"])
    # This is the critical assertion: (:v,) is a tuple, (:v) is just the symbol
    assert "(:v,)" in code
    assert "(:GNa,)" in code
    assert "(:Calcium,)" in code
    # Ensure it's not the bare symbol without comma
    lines = code.split("\n")
    for line in lines:
        if ":v" in line:
            assert ":v," in line, "Single-element state name must have trailing comma"

    # Multi-element list case: all names in order, trailing comma after last
    code = cytozoo.name_tuples("ORd", ["v", "cai", "cass"], ["GNa", "GK"], ["ICa", "IK"])
    assert "(:v, :cai, :cass,)" in code
    assert "(:GNa, :GK,)" in code
    assert "(:ICa, :IK,)" in code


def test_interface_methods_conditional_monitor_branch():
    # With monitors: all three required components must be present
    code = cytozoo.interface_methods("ORdmmLand", 0, 3)
    assert "num_monitors(::ORdmmLand) = 3" in code
    assert "monitor_names(::ORdmmLand) = ORDMMLAND_MONITOR_NAMES" in code
    assert "function monitor_values!(mon, u, t, model::ORdmmLand)" in code
    # The critical detail: monitor_values must be called with six arguments
    # ending with mon, nothing, nothing (spatial context additions come later)
    assert "monitor_values(t, u, model.parameters, mon, nothing, nothing)" in code

    # Without monitors: none of those three should appear
    code = cytozoo.interface_methods("ORdmmLand", 0, 0)
    assert "num_monitors" not in code
    assert "monitor_names" not in code
    assert "function monitor_values!" not in code

    # Test with v_index set (should appear in both cases)
    code = cytozoo.interface_methods("ORdmmLand", 42, 0)
    assert "transmembrane_potential_index(::ORdmmLand) = 42" in code


import pytest
from gotranx.codegen.cytozoo import (
    CytoZooCodeGenerator,
    InvalidModelName,
    MissingPotentialState,
    _pascal_case,
)
from gotranx.ode import make_ode


# Kept as one constant so the fixture and the two CLI tests below cannot drift.
SIMPLE_ODE = """
parameters(GNa=1.0, T=310.0)
states(v=-87.0, ca=1e-4)

dv_dt = -GNa*v
dca_dt = T*ca
"""


@pytest.fixture(scope="module")
def simple_ode(parser, trans):
    tree = parser.parse(SIMPLE_ODE)
    return make_ode(*trans.transform(tree), name="simple")


def test_adapter_names_the_model_from_the_ode(simple_ode):
    code = CytoZooCodeGenerator(simple_ode, type_stable=True).adapter()
    assert "struct Simple{T <: AbstractVector} <: AbstractCardiacCellModel" in code


def test_transmembrane_potential_index_is_one_based(simple_ode):
    code = CytoZooCodeGenerator(simple_ode, type_stable=True).adapter()
    idx = [s.name for s in simple_ode.states].index("v") + 1
    assert f"transmembrane_potential_index(::Simple) = {idx}" in code


def test_missing_potential_state_fails_loudly(parser, trans):
    tree = parser.parse(
        """
        parameters(a=1.0)
        states(x=0.0)

        dx_dt = -a*x
        """
    )
    ode = make_ode(*trans.transform(tree), name="nov")
    with pytest.raises(MissingPotentialState) as exc:
        CytoZooCodeGenerator(ode, type_stable=True).adapter()
    # The error must name the option to set, not merely complain.
    assert "v_name" in str(exc.value)


@pytest.mark.parametrize(
    "ode_name",
    [
        # A leading digit: `1962_noble` pascal-cases to `1962Noble`.
        "1962_noble",
        # A space survives `_pascal_case`, which only splits on `-` and `_`.
        "tentusscher panfilov",
        # So does a dot, which Julia reads as module qualification.
        "my.model",
    ],
)
def test_a_model_name_that_is_not_an_identifier_is_rejected(parser, trans, ode_name):
    """Each of these used to emit `struct <invalid>{...}`: a Julia parse error."""
    tree = parser.parse("parameters(a=1.0)\nstates(v=0.0)\n\ndv_dt = -a*v\n")
    ode = make_ode(*trans.transform(tree), name=ode_name)
    with pytest.raises(InvalidModelName) as exc:
        CytoZooCodeGenerator(ode, type_stable=True)
    # The message must name the offending value, not merely complain.
    assert _pascal_case(ode_name) in str(exc.value)
    assert "--model-name" in str(exc.value)


def test_an_explicit_model_name_is_validated_too(simple_ode):
    with pytest.raises(InvalidModelName) as exc:
        CytoZooCodeGenerator(simple_ode, model_name="1962Noble", type_stable=True)
    assert "1962Noble" in str(exc.value)


def test_both_cytozoo_exceptions_are_gotranx_errors():
    """`except exceptions.GotranxError` must catch them, like the other twelve."""
    from gotranx import exceptions

    assert issubclass(exceptions.InvalidModelName, exceptions.GotranxError)
    assert issubclass(exceptions.MissingPotentialState, exceptions.GotranxError)
    # And they are still importable from where they are raised.
    assert InvalidModelName is exceptions.InvalidModelName
    assert MissingPotentialState is exceptions.MissingPotentialState


def test_state_names_follow_the_ode_ordering(simple_ode):
    code = CytoZooCodeGenerator(simple_ode, type_stable=True).adapter()
    expected = ", ".join(f":{s.name}" for s in simple_ode.states)
    assert expected in code


from gotranx import atoms


@pytest.fixture(scope="module")
def diverging_ode(parser, trans):
    # `a`'s derivative depends on a two-level intermediate chain rooted at
    # `z`; `z`'s derivative depends on nothing but itself and a parameter.
    # Named for the two ends of the alphabet so that alphabetical order (a, z)
    # and dependency order (z must be resolved before the chain that feeds a)
    # disagree -- the property the two tests below rely on, and which each of
    # them re-asserts rather than assuming.
    tree = parser.parse(
        """
        parameters(k=1.0)
        states(a=0.0, z=1.0)

        chain1 = z
        chain2 = chain1 * 2
        da_dt = chain2 - a
        dz_dt = -k*z
        """
    )
    return make_ode(*trans.transform(tree), name="diverge")


def test_transmembrane_potential_index_uses_dependency_order(diverging_ode):
    alphabetical = [s.name for s in diverging_ode.states]
    dependency_order = [s.name for s in diverging_ode.sorted_states()]
    # If this ever stops diverging the test below stops testing anything --
    # fail loudly rather than silently passing for the wrong reason.
    assert alphabetical.index("z") != dependency_order.index("z")

    code = CytoZooCodeGenerator(diverging_ode, v_name="z", type_stable=True).adapter()
    expected_index = dependency_order.index("z") + 1
    assert f"transmembrane_potential_index(::Diverge) = {expected_index}" in code


def test_monitor_names_follow_monitor_index_order(diverging_ode):
    # Mirror codegen.base.CodeGenerator.monitor_index's own loop exactly: walk
    # sorted_assignments() and keep Intermediate/StateDerivative names in the
    # single interleaved order they're encountered in, not grouped by type.
    expected = [
        a.name
        for a in diverging_ode.sorted_assignments(remove_unused=False)
        if isinstance(a, (atoms.Intermediate, atoms.StateDerivative))
    ]
    # If the two kinds never interleave, this test can't catch a grouped-by-type bug.
    kinds = [
        isinstance(a, atoms.StateDerivative)
        for a in diverging_ode.sorted_assignments(remove_unused=False)
        if isinstance(a, (atoms.Intermediate, atoms.StateDerivative))
    ]
    assert kinds != sorted(kinds), "fixture no longer interleaves intermediates and derivatives"

    code = CytoZooCodeGenerator(diverging_ode, v_name="z", type_stable=True).adapter()
    expected_tuple = "(" + ", ".join(f":{n}" for n in expected) + ",)"
    assert f"const DIVERGE_MONITOR_NAMES = {expected_tuple}" in code


def test_rhs_takes_spatial_arguments(simple_ode):
    rhs = CytoZooCodeGenerator(simple_ode, type_stable=True).rhs()
    assert "_cz_x" in rhs
    assert "_cz_overrides" in rhs


def test_every_parameter_is_spatially_overridable(simple_ode):
    rhs = CytoZooCodeGenerator(simple_ode, type_stable=True).rhs()
    for p in simple_ode.parameters:
        assert f"Val(:{p.name})" in rhs, f"{p.name} is not overridable"


def test_parameter_named_T_does_not_shadow_the_type_parameter(simple_ode):
    # simple_ode has a parameter literally named T. CytoZoo's docs name the element
    # type T; gotranx names it TYPE. Emitting both would produce `T = parameters[2]`
    # followed by `T(0.5)` -- calling a Float64.
    rhs = CytoZooCodeGenerator(simple_ode, type_stable=True).rhs()
    assert "where {TYPE" in rhs
    assert "where {T," not in rhs
    assert "where {T}" not in rhs
    assert "T = resolve_parameter" in rhs or "T = TYPE(resolve_parameter" in rhs


def test_the_overrides_type_parameter_cannot_collide_with_a_parameter(parser, trans):
    """CytoZoo's guide spells it `F`; `F` is Faraday's constant in every cardiac model.

    `function rhs(..., _cz_overrides::F) where {..., F}` with a body that then
    assigns `F = TYPE(resolve_parameter(...))` is a Julia *syntax* error --
    "local variable name \"F\" conflicts with a static parameter" -- so the
    module does not load at all. ORdmm_Land has exactly such a parameter.
    """
    tree = parser.parse(
        """
        parameters(F=96485.0)
        states(v=-87.0)

        dv_dt = -v/F
        """
    )
    ode = make_ode(*trans.transform(tree), name="faraday")
    rhs = CytoZooCodeGenerator(ode, type_stable=True).rhs()

    # The parameter local is still emitted, and still named F.
    assert "F = TYPE(resolve_parameter(" in rhs
    # The overrides' type parameter must therefore not be called F.
    signature = next(line for line in rhs.splitlines() if line.startswith("function rhs("))
    assert "_cz_overrides::_cz_F" in signature
    assert re.search(r"where \{[^}]*\bF\b", signature) is None, signature


def test_plain_julia_backend_is_untouched(simple_ode):
    from gotranx.codegen import JuliaCodeGenerator

    rhs = JuliaCodeGenerator(simple_ode, type_stable=True).rhs()
    assert "_cz_overrides" not in rhs
    assert "resolve_parameter" not in rhs


@pytest.fixture(scope="module")
def permuted_ode(parser, trans):
    # Declared out of alphabetical order, and the declaration order itself is
    # neither the sorted order nor its reversal, so neither a sort-vs-declaration
    # mixup nor a reversal would coincidentally line up with the correct pairing.
    tree = parser.parse(
        """
        parameters(Omega=1.0, Gamma=2.0, Sigma=3.0, Delta=4.0)
        states(v=-87.0)

        dv_dt = -(Omega + Gamma + Sigma + Delta) * v
        """
    )
    return make_ode(*trans.transform(tree), name="permuted")


def test_parameter_name_index_pairing_matches_ode_order(permuted_ode):
    # The four tests above only assert that tokens like `Val(:GNa)` appear
    # *somewhere* in the output. A bug that scrambles the pairing between a
    # parameter's name and its `parameters[i]` index -- e.g. enumerating
    # `reversed(self.ode.parameters)` while still printing the right name --
    # would leave every one of those tokens present and still pass. Only
    # checking the actual (name, index) pairs against the ODE's own ordering
    # catches that silent mislabeling.
    gen = CytoZooCodeGenerator(permuted_ode, type_stable=True)
    rhs = gen.rhs()

    pairs = re.findall(r"(\w+) = TYPE\(resolve_parameter\(parameters\[(\d+)\]", rhs)
    index_by_name = {name: int(idx) for name, idx in pairs}

    expected_names = [p.name for p in permuted_ode.parameters]
    assert len(index_by_name) == len(expected_names) == 4
    for i, name in enumerate(expected_names):
        assert index_by_name[name] == i + 1, (
            f"{name} should read parameters[{i + 1}], got parameters[{index_by_name[name]}]"
        )

    # Pin the rhs assignments and the adapter's PARAMETER_NAMES tuple against
    # each other -- not each independently against a literal -- so the two
    # halves of the name<->index contract can't silently drift apart.
    adapter_code = gen.adapter()
    ordered_from_rhs = [name for name, _ in sorted(index_by_name.items(), key=lambda kv: kv[1])]
    expected_tuple = "(" + ", ".join(f":{n}" for n in ordered_from_rhs) + ",)"
    assert f"PERMUTED_PARAMETER_NAMES = {expected_tuple}" in adapter_code


def test_monitor_hooks_are_emitted(simple_ode):
    code = CytoZooCodeGenerator(simple_ode, type_stable=True).adapter()
    assert "num_monitors(::Simple) = " in code
    assert "monitor_names(::Simple) = SIMPLE_MONITOR_NAMES" in code
    assert "function monitor_values!(mon, u, t, model::Simple)" in code


def test_an_ode_with_no_monitors_emits_no_monitor_block(parser, trans):
    # Only state derivatives, no intermediates -> monitors still exist; force the
    # zero case directly, since CytoZoo's defaults must carry it.
    from gotranx.templates import cytozoo as tpl

    code = tpl.interface_methods("Bare", 1, 0)
    assert "num_monitors" not in code
    assert "monitor_values!" not in code


def test_monitor_values_takes_spatial_arguments(simple_ode):
    code = CytoZooCodeGenerator(simple_ode, type_stable=True).monitor_values()
    assert "_cz_overrides" in code
    # `"_cz_overrides" in code` would still pass if the name moved anywhere in
    # the body. Pin the definition's own argument list, mirroring the call-site
    # assertion in test_interface_methods_conditional_monitor_branch: the
    # emitted `monitor_values!` adapter calls it with six positional arguments
    # ending `nothing, nothing`, so the definition must end with exactly these
    # two in this order.
    assert re.search(r"function monitor_values\([^)]*,\s*_cz_x,\s*_cz_overrides::_cz_F\)", code), (
        code.splitlines()[0]
    )


def test_cli_writes_a_module(tmp_path):
    from typer.testing import CliRunner
    from gotranx.cli import app

    ode = tmp_path / "simple.ode"
    ode.write_text(SIMPLE_ODE)
    result = CliRunner().invoke(app, ["ode2cytozoo", str(ode), "-o", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output

    code = (tmp_path / "out.jl").read_text()
    assert "import CytoZoo" in code
    assert "AbstractCardiacCellModel" in code
    assert "resolve_parameter" in code


def test_ode2julia_output_is_unchanged_by_this_feature(tmp_path):
    """The root repo's generate.jl --check compares bytes. This must never move."""
    from gotranx.cli import gotran2julia
    from gotranx.load import load_ode

    ode_file = tmp_path / "m.ode"
    ode_file.write_text(SIMPLE_ODE)
    code = gotran2julia.get_code(load_ode(ode_file), type_stable=True)
    assert "resolve_parameter" not in code
    assert "_cz_" not in code
    assert "AbstractCardiacCellModel" not in code


def test_extended_cytozoo_interface_names_are_imported_not_used(simple_ode):
    """Julia requires `import Mod: f` (not `using Mod: f`) to add methods to `f`.

    Every CytoZoo function the generated adapter defines a new method on must be
    `import`ed, or the module fails to load with "function M.f must be explicitly
    imported to be extended" -- a load-time error a textual grep for a name can't
    catch, since the name is present either way. This test derives the set of
    extended names from the generated code itself (every `name(::Simple...)` or
    `function name(..., model::Simple...)` definition) rather than hard-coding a
    second copy of the interface list, so it can't silently drift from
    templates/cytozoo.py.
    """
    import re
    from gotranx.cli import gotran2cytozoo

    code = gotran2cytozoo.get_code(simple_ode)
    model_name = "Simple"

    extended = sorted(
        set(
            re.findall(
                rf"(?:^|\n)\s*(?:function\s+)?(\w+!?)\([^)\n]*::{model_name}\b",
                code,
            )
        )
    )
    # If the fixture ever stops emitting any method on the model type, this test
    # would vacuously pass without checking anything -- fail loudly instead.
    assert extended, "no extended interface names found; fixture no longer exercises this"

    import_names: set[str] = set()
    using_names: set[str] = set()
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import CytoZoo:"):
            import_names.update(n.strip() for n in stripped.split(":", 1)[1].split(","))
        elif stripped.startswith("using CytoZoo:"):
            using_names.update(n.strip() for n in stripped.split(":", 1)[1].split(","))

    for name in extended:
        assert name in import_names, f"{name} is extended but not on an `import CytoZoo:` line"
        assert name not in using_names, f"{name} is extended but also brought in via `using`"


_CYTOZOO_SCRIPT = """
import sys
import gotranx

ode = gotranx.load_ode(sys.argv[1])
from gotranx.cli.gotran2cytozoo import get_code
sys.stdout.write("<<<CODE>>>" + get_code(ode))
"""

SEEDS = ("1", "2", "7")


def _run(script: str, odefile: Path, seed: str, marker: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    out = subprocess.run(
        [sys.executable, "-c", script, str(odefile)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert marker in out.stdout, f"marker missing; stderr was:\n{out.stderr}"
    return out.stdout.split(marker, 1)[1]


def test_generation_is_deterministic():
    """The consuming repo's regenerate-and-compare CI check depends on this.

    PYTHONHASHSEED is fixed per interpreter, so in-process comparisons cannot
    detect hash-seed-dependent ordering bugs. This test shells out to fresh
    interpreters with different seeds, following test_determinism.py's pattern.
    """
    odefile = Path(__file__).parent / "odefiles" / "ORdmm_Land.ode"
    code = [_run(_CYTOZOO_SCRIPT, odefile, s, "<<<CODE>>>") for s in SEEDS]

    for seed, other in zip(SEEDS[1:], code[1:]):
        assert other == code[0], (
            f"generated code under PYTHONHASHSEED={seed} differs from {SEEDS[0]}"
        )

    # Also verify in-process determinism: get_code is a pure function of its input.
    from gotranx.cli.gotran2cytozoo import get_code
    from gotranx.load import load_ode

    ode = load_ode(odefile)
    assert get_code(ode) == get_code(load_ode(odefile))


def test_real_model_generates():
    from gotranx.cli.gotran2cytozoo import get_code
    from gotranx.load import load_ode

    ode = load_ode(Path(__file__).parent / "odefiles" / "ORdmm_Land.ode")
    code = get_code(ode)
    # The type parameter must be named TYPE, not T (which is a parameter name).
    assert "where {TYPE" in code
    # The parameter named T must be overridable via Val(:T) (secondary check).
    assert "Val(:T)" in code
    # The adapter must include the transmembrane potential index.
    assert "transmembrane_potential_index" in code


def test_generated_module_carries_a_provenance_banner(simple_ode):
    """Hand-written headers are deleted the moment anyone regenerates the file.

    The banner has to come out of the generator, so the facts a reader needs --
    that the file is generated, by which version, from what, and what it
    requires -- survive a regeneration.
    """
    import gotranx
    from gotranx.cli.gotran2cytozoo import get_code

    code = get_code(simple_ode, source="/somewhere/on/my/disk/simple.ode")
    banner = code.split("import CytoZoo", 1)[0]

    assert banner.strip(), "no banner before the first import"
    assert all(line.startswith("#") for line in banner.strip().splitlines())

    # The version is read from the package, never hardcoded.
    assert gotranx.__version__ in banner
    assert "gotranx" in banner
    assert "do not hand-edit" in banner

    # The basename only: an absolute path would leak one machine's layout into
    # a file that gets committed in the consuming repository.
    assert "simple.ode" in banner
    assert "/somewhere/on/my/disk" not in code

    # The two facts a reader cannot get from the code itself.
    assert "resolve_parameter" in banner
    assert "module" in banner


def test_the_banner_falls_back_to_the_ode_name(simple_ode):
    from gotranx.cli.gotran2cytozoo import get_code

    banner = get_code(simple_ode).split("import CytoZoo", 1)[0]
    assert f"{simple_ode.name}.ode" in banner
