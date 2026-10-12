from __future__ import annotations
from textwrap import dedent


def model_struct(name: str, num_params: int, num_missing: int = 0) -> str:
    """The model struct and its three constructors.

    One field, unlike ToRORd's three: cell type and stimulus settings are ordinary
    parameters in a .ode, so they live in the flat vector. Naming the field
    `parameters` is load-bearing — CytoZoo's `writable_parameters` default returns
    `model.parameters`, which is what a `connect` edge writes into.

    A model with missing variables (a component subtracted away) keeps them in the
    same vector, after its parameters, so a `connect` edge supplies them by name.
    They start as NaN: an input nothing has been connected to poisons the RHS
    instead of silently reading 0.
    """
    size = num_params + num_missing
    # Indented like the template's function bodies below, so that `dedent` still finds
    # their common prefix.
    inputs = "\n" + " " * 12 + "p[(NUM_PARAMS + 1):end] .= NaN" if num_missing else ""
    return dedent(
        f"""
        struct {name}{{T <: AbstractVector}} <: AbstractCardiacCellModel
            parameters::T
        end

        {name}() = {name}(Float64)

        function {name}(::Type{{ElT}}) where {{ElT <: Number}}
            p = zeros(ElT, {size})
            init_parameter_values!(p){inputs}
            return {name}(p)
        end

        function {name}(::Type{{VT}}) where {{VT <: AbstractVector}}
            p = zeros(eltype(VT), {size})
            init_parameter_values!(p){inputs}
            return {name}(VT(p))
        end
        """
    )


def name_tuples(
    name: str,
    state_names: list[str],
    parameter_names: list[str],
    monitor_names: list[str],
) -> str:
    def tup(names: list[str]) -> str:
        if not names:
            return "()"
        return "(" + ", ".join(f":{n}" for n in names) + ",)"

    return dedent(
        f"""
        const {name.upper()}_STATE_NAMES = {tup(state_names)}
        const {name.upper()}_PARAMETER_NAMES = {tup(parameter_names)}
        const {name.upper()}_MONITOR_NAMES = {tup(monitor_names)}
        """
    )


def index_lookups(name: str, state_names: list[str], parameter_names: list[str]) -> str:
    """Symbol-keyed lookups that RAISE on an unknown name.

    gotranx's own `state_index(name::String)` returns -1 for an unknown name. Passing
    that through would make a typo into `u[-1]`: a BoundsError far from its cause, or a
    silently wrong read. A Dict raises KeyError at the point of the mistake.
    """
    upper = name.upper()
    return dedent(
        f"""
        const {upper}_STATE_INDEX = Dict{{Symbol, Int}}(
            n => i for (i, n) in enumerate({upper}_STATE_NAMES)
        )
        const {upper}_PARAMETER_INDEX = Dict{{Symbol, Int}}(
            n => i for (i, n) in enumerate({upper}_PARAMETER_NAMES)
        )

        state_index(::{name}, name::Symbol) = {upper}_STATE_INDEX[name]
        parameter_index(::{name}, name::Symbol) = {upper}_PARAMETER_INDEX[name]
        state_names(::{name}) = {upper}_STATE_NAMES
        parameter_names(::{name}) = {upper}_PARAMETER_NAMES
        """
    )


def interface_methods(name: str, v_index: int, num_monitors: int, num_missing: int = 0) -> str:
    upper = name.upper()
    # The missing variables are the trailing slice of the parameter vector (see model_struct).
    inputs = (
        ", view(model.parameters, (NUM_PARAMS + 1):(NUM_PARAMS + NUM_MISSING))"
        if num_missing
        else ""
    )
    num_parameters = "NUM_PARAMS + NUM_MISSING" if num_missing else "NUM_PARAMS"
    count = "Seven" if num_missing else "Six"
    missing_comment = (
        "\n" + " " * 16 + "# The seventh is the missing variables." if num_missing else ""
    )
    monitors = ""
    if num_monitors > 0:
        monitors = dedent(
            f"""
            num_monitors(::{name}) = {num_monitors}
            monitor_names(::{name}) = {upper}_MONITOR_NAMES

            function monitor_values!(mon, u, t, model::{name})
                # {count} arguments: monitor_values inherits the two spatial arguments
                # from _rhs_arguments, exactly as rhs does. Monitors are not
                # spatially overridden, so both are nothing.{missing_comment}
                monitor_values(t, u, model.parameters, mon, nothing, nothing{inputs})
                return nothing
            end
            """
        )

    core = dedent(
        f"""
        num_states(::{name}) = NUM_STATES
        num_parameters(::{name}) = {num_parameters}
        transmembrane_potential_index(::{name}) = {v_index}

        function default_initial_state(model::{name}{{T}}) where {{T}}
            u = zeros(eltype(T), NUM_STATES)
            init_state_values!(u)
            return u
        end

        function (model::{name})(du, u, ::Nothing, t)
            rhs(t, u, model.parameters, du, nothing, nothing{inputs})
            return nothing
        end

        function (model::{name})(du, u, p::SpatialContext, t)
            rhs(t, u, model.parameters, du, p.x, p.overrides{inputs})
            return nothing
        end
        """
    )
    return core + monitors
