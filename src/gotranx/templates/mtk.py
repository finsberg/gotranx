from __future__ import annotations

from textwrap import dedent, indent


def header() -> str:
    return dedent(
        """
        using ModelingToolkit
        """,
    )


def parameters_block(params: list[str], extra_params: list[str]) -> str:
    names = " ".join(["t"] + params + extra_params)
    return dedent(
        f"""
        @parameters {names}
        D = Differential(t)
        """,
    )


def states_block(states: list[str]) -> str:
    names = " ".join([f"{s}(t)" for s in states])
    return dedent(
        f"""
        @variables {names}
        """,
    )


def observed_block(entries: list[str]) -> str:
    if not entries:
        return "observed = []"

    body = indent(",\n".join(entries), "    ")
    return dedent(
        f"""
        observed = [
        {body}
        ]
        """,
    )


def equations_block(entries: list[str]) -> str:
    body = indent(",\n".join(entries), "    ")
    return dedent(
        f"""
        eqs = [
        {body}
        ]
        """,
    )


def defaults_block(
    state_defaults: list[tuple[str, str]], param_defaults: list[tuple[str, str]]
) -> str:
    pairs = []
    for name, value in state_defaults + param_defaults:
        pairs.append(f"{name} => {value}")

    if not pairs:
        return "defaults = Dict()"

    body = indent(",\n".join(pairs), "    ")
    return dedent(
        f"""
        defaults = Dict(
        {body}
        )
        """,
    )


def ode_system(name: str, state_names: list[str], param_names: list[str]) -> str:
    states = ", ".join(state_names)
    params = ", ".join(param_names)
    return dedent(
        f"""
        @named {name} = ODESystem(
            eqs,
            t,
            [{states}],
            [{params}],
            observed = observed,
            defaults = defaults,
        )
        """,
    )
