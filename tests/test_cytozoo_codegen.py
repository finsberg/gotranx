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
from gotranx.codegen.cytozoo import CytoZooCodeGenerator, MissingPotentialState
from gotranx.ode import make_ode


@pytest.fixture(scope="module")
def simple_ode(parser, trans):
    tree = parser.parse(
        """
        parameters(GNa=1.0, T=310.0)
        states(v=-87.0, cai=1e-4)

        dv_dt = -GNa*v
        dcai_dt = T*cai
        """
    )
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


def test_state_names_follow_the_ode_ordering(simple_ode):
    code = CytoZooCodeGenerator(simple_ode, type_stable=True).adapter()
    expected = ", ".join(f":{s.name}" for s in simple_ode.states)
    assert expected in code


from gotranx import atoms


@pytest.fixture(scope="module")
def diverging_ode(parser, trans):
    # `alpha`'s derivative depends on a two-level intermediate chain rooted at
    # `zeta`; `zeta`'s derivative depends on nothing but itself and a parameter.
    # Named so that alphabetical order (alpha, zeta) and dependency order
    # (zeta must be resolved before the chain that feeds alpha) disagree.
    tree = parser.parse(
        """
        parameters(k=1.0)
        states(alpha=0.0, zeta=1.0)

        chain1 = zeta
        chain2 = chain1 * 2
        dalpha_dt = chain2 - alpha
        dzeta_dt = -k*zeta
        """
    )
    return make_ode(*trans.transform(tree), name="diverge")


def test_transmembrane_potential_index_uses_dependency_order(diverging_ode):
    alphabetical = [s.name for s in diverging_ode.states]
    dependency_order = [s.name for s in diverging_ode.sorted_states()]
    # If this ever stops diverging the test below stops testing anything --
    # fail loudly rather than silently passing for the wrong reason.
    assert alphabetical.index("zeta") != dependency_order.index("zeta")

    code = CytoZooCodeGenerator(diverging_ode, v_name="zeta", type_stable=True).adapter()
    expected_index = dependency_order.index("zeta") + 1
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

    code = CytoZooCodeGenerator(diverging_ode, v_name="zeta", type_stable=True).adapter()
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


def test_plain_julia_backend_is_untouched(simple_ode):
    from gotranx.codegen import JuliaCodeGenerator

    rhs = JuliaCodeGenerator(simple_ode, type_stable=True).rhs()
    assert "_cz_overrides" not in rhs
    assert "resolve_parameter" not in rhs
