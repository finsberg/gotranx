from __future__ import annotations
from collections import defaultdict

from functools import cached_property
from pathlib import Path
from graphlib import TopologicalSorter
from typing import Iterable
from typing import Sequence
from typing import TypeVar
from typing import cast
from typing import Any

import sympy as sp

from . import atoms
from . import exceptions
from .ode_component import Component

T = TypeVar("T")
U = TypeVar("U", bound=atoms.Assignment)


def check_components(components: Sequence[Component]):
    for comp in components:
        if not comp.is_complete():
            raise exceptions.ComponentNotCompleteError(
                component_name=comp.name,
                missing_state_derivatives=[state.name for state in comp.states_without_derivatives],
            )


def gather_atoms(
    components: Sequence[Component],
) -> tuple[list[str], dict[str, set[Any]], dict[str, sp.Symbol], dict[str, atoms.Atom]]:
    symbol_names = []
    symbols = {}
    symbol_values = defaultdict(set)
    lookup: dict[str, atoms.Atom] = {}
    for component in components:
        for p in component.parameters:
            symbol_names.append(p.name)
            symbols[p.name] = p.symbol
            lookup[p.name] = p
            symbol_values[p.name].add(p.value)
        for s in component.states:
            symbol_names.append(s.name)
            symbols[s.name] = s.symbol
            lookup[s.name] = s
            symbol_values[s.name].add(s.value)
        for i in component.intermediates:
            symbol_names.append(i.name)
            symbols[i.name] = i.symbol
            lookup[i.name] = i
            symbol_values[i.name].add(i.expr)
        for st in component.state_derivatives:
            symbol_names.append(st.name)
            symbols[st.name] = st.symbol
            lookup[st.name] = st
    return symbol_names, symbol_values, symbols, lookup


def find_duplicates(x: Iterable[T]) -> set[T]:
    """Find duplicates in an iterable.
    Assumes type is hashable

    Parameters
    ----------
    x : Iterable[T]
        The iterable with potential duplicates

    Returns
    -------
    set[T]
        List of duplicate values
    """
    seen = set()
    dupes = []

    for xi in x:
        if xi in seen:
            dupes.append(xi)
        else:
            seen.add(xi)
    return set(dupes)


def add_temporal_state(
    components: Sequence[Component],
    t: sp.Symbol,
) -> tuple[Component, ...]:
    new_components = []
    for component in components:
        state_derivatives = set()
        states = set()
        for state_derivative in component.state_derivatives:
            new_state = state_derivative.state.to_TimeDependentState(t)
            state_derivatives.add(
                atoms.StateDerivative(
                    name=state_derivative.name,
                    value=state_derivative.value,
                    components=state_derivative.components,
                    description=state_derivative.description,
                    unit_str=state_derivative.unit_str,
                    unit=state_derivative.unit,
                    state=new_state,
                    symbol=sp.Derivative(new_state.symbol, t),
                ),
            )
            states.add(new_state)

        new_components.append(
            Component(
                name=component.name,
                states=frozenset(states),
                parameters=component.parameters,
                assignments=frozenset(component.intermediates | state_derivatives),
            ),
        )
    return tuple(new_components)


def resolve_expressions(
    components: Sequence[Component],
    symbols: dict[str, sp.Symbol],
) -> tuple[Component, ...]:
    new_components = []
    for component in components:
        assignments = []
        for assignment in component.assignments:
            assignments.append(assignment.resolve_expression(symbols))
        new_components.append(
            Component(
                name=component.name,
                states=component.states,
                parameters=component.parameters,
                assignments=frozenset(assignments),
            ),
        )
    return tuple(new_components)


def make_ode(
    components: Sequence[Component],
    comments: Sequence[atoms.Comment] | None = None,
    name: str = "ODE",
) -> ODE:
    check_components(components=components)
    t = sp.Symbol("t")
    # components = add_temporal_state(components, t)
    check_components(components=components)
    _, symbol_values, symbols, lookup = gather_atoms(components=components)
    symbols["time"] = t
    symbols["t"] = t

    if any(x > 1 for x in map(len, symbol_values.values())):
        raise exceptions.DuplicateSymbolError(
            set(k for k, v in symbol_values.items() if len(v) > 1)
        )
    components = resolve_expressions(components=components, symbols=symbols)
    return ODE(components=components, t=t, name=name, comments=comments)


def sort_assignments(
    assignments: Iterable[atoms.Assignment],
    assignments_only: bool = True,
) -> tuple[str, ...]:
    sorter: TopologicalSorter = TopologicalSorter()
    assignment_names = set()
    for assignment in assignments:
        assignment_names.add(assignment.name)
        sorter.add(assignment.name, *assignment.value.dependencies)

    static_order = tuple(sorter.static_order())

    if assignments_only:
        static_order = tuple([name for name in static_order if name in assignment_names])

    return static_order


class ODE:
    def __init__(
        self,
        components: Sequence[Component],
        t: sp.Symbol | None = None,
        name: str = "ODE",
        comments: Sequence[atoms.Comment] | None = None,
    ):
        check_components(components)
        _, symbol_values, symbols, lookup = gather_atoms(components=components)

        if any(x > 1 for x in map(len, symbol_values.values())):
            raise exceptions.DuplicateSymbolError(
                set(k for k, v in symbol_values.items() if len(v) > 1)
            )

        t = sp.Symbol("t")
        self.t = t
        symbols["time"] = t

        self._symbols = symbols
        self._lookup = lookup
        self.components = components
        self.name = name
        if comments is None:
            comments = (atoms.Comment(""),)

        self.comments = comments
        self.text = "".join(comment.text for comment in comments)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.name}, "
            f"num_states={self.num_states}, "
            f"num_parameters={self.num_parameters})"
        )

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ODE):
            return False

        return (
            __o.comments == self.comments
            and __o.components == self.components
            and __o.name == self.name
        )

    def simplify(self) -> ODE:
        """Run sympy's simplify function on all expressions in the ODE"""
        return ODE(
            components=tuple([comp.simplify() for comp in self.components]),
            t=self.t,
            name=self.name,
            comments=self.comments,
        )

    @property
    def states(self) -> tuple[atoms.State, ...]:
        states: set[atoms.State] = set()
        for component in self.components:
            states |= component.states
        return tuple(sorted(states, key=lambda x: x.name))

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def num_components(self) -> int:
        return len(self.components)

    @cached_property
    def parameters(self) -> tuple[atoms.Parameter, ...]:
        parameters: set[atoms.Parameter] = set()
        for component in self.components:
            parameters |= component.parameters
        return tuple(sorted(parameters, key=lambda x: x.name))

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    @cached_property
    def state_derivatives(self) -> tuple[atoms.StateDerivative, ...]:
        state_derivatives: set[atoms.StateDerivative] = set()
        for component in self.components:
            state_derivatives |= component.state_derivatives
        return tuple(sorted(state_derivatives, key=lambda x: x.name))

    @cached_property
    def intermediates(self) -> tuple[atoms.Intermediate, ...]:
        intermediates: set[atoms.Intermediate] = set()
        for component in self.components:
            intermediates |= component.intermediates
        return tuple(sorted(intermediates, key=lambda x: x.name))

    @property
    def symbols(self) -> dict[str, sp.Symbol]:
        return self._symbols

    def __getitem__(self, name) -> atoms.Atom:
        return self._lookup[name]

    def sorted_assignments(self, assignments_only: bool = True) -> tuple[atoms.Assignment, ...]:
        names = sort_assignments(
            self.intermediates + self.state_derivatives,
            assignments_only=assignments_only,
        )
        return tuple([cast(atoms.Assignment, self[name]) for name in names])

    def sorted_state_derivatives(self) -> tuple[atoms.StateDerivative, ...]:
        return tuple(s for s in self.sorted_assignments() if isinstance(s, atoms.StateDerivative))

    def sorted_states(self) -> tuple[atoms.State, ...]:
        return tuple(s.state for s in self.sorted_state_derivatives())

    def save(self, path: Path) -> None:
        from .save import write_ODE_to_ode_file

        write_ODE_to_ode_file(self, path)
