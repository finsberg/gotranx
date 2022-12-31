from __future__ import annotations

from functools import cached_property
from graphlib import TopologicalSorter
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import TypeVar

import sympy as sp

from . import atoms
from . import exceptions
from .ode_component import Component

T = TypeVar("T")


def check_components(components: Sequence[Component]):
    for comp in components:
        if not comp.is_complete():
            raise exceptions.ComponentNotCompleteError(
                component_name=comp.name,
                missing_state_derivatives=[
                    state.name for state in comp.states_without_derivatives
                ],
            )


def gather_atoms(
    components: Sequence[Component],
) -> tuple[list[str], dict[str, sp.Symbol], dict[str, atoms.Atom]]:
    symbol_names = []
    symbols = {}
    lookup: dict[str, atoms.Atom] = {}
    for component in components:
        for p in component.parameters:
            symbol_names.append(p.name)
            symbols[p.name] = p.symbol
            lookup[p.name] = p
        for s in component.states:
            symbol_names.append(s.name)
            symbols[s.name] = s.symbol
            lookup[s.name] = s
        for a in component.assignments:
            symbol_names.append(a.name)
            symbols[a.name] = a.symbol
            lookup[a.name] = a
    return symbol_names, symbols, lookup


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
                    component=state_derivative.component,
                    description=state_derivative.description,
                    info=state_derivative.info,
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
    comments: Optional[Sequence[atoms.Comment]] = None,
    name: str = "ODE",
) -> ODE:
    check_components(components=components)
    t = sp.Symbol("t")
    # components = add_temporal_state(components, t)
    check_components(components=components)
    symbol_names, symbols, lookup = gather_atoms(components=components)
    symbols["time"] = t

    if not len(symbol_names) == len(set(symbol_names)):
        raise exceptions.DuplicateSymbolError(find_duplicates(symbol_names))
    components = resolve_expressions(components=components, symbols=symbols)
    return ODE(components=components, t=t, name=name, comments=comments)


def sort_assignments(
    assignments: Iterable[atoms.Assignment],
    assignments_only: bool = False,
) -> tuple[str, ...]:
    sorter: TopologicalSorter = TopologicalSorter()
    assignment_names = set()
    for assignment in assignments:
        assignment_names.add(assignment.name)
        sorter.add(assignment.name, *assignment.value.dependencies)

    static_order = tuple(sorter.static_order())

    if assignments_only:
        return tuple([name for name in static_order if name in assignment_names])
    else:
        return static_order


class ODE:
    def __init__(
        self,
        components: Sequence[Component],
        t: Optional[sp.Symbol] = None,
        name: str = "ODE",
        comments: Optional[Sequence[atoms.Comment]] = None,
    ):

        check_components(components)
        symbol_names, symbols, lookup = gather_atoms(components=components)
        if not len(symbol_names) == len(set(symbol_names)):
            raise exceptions.DuplicateSymbolError(find_duplicates(symbol_names))
        if t is None:
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

    @cached_property
    def states(self) -> frozenset[atoms.State]:
        states: set[atoms.State] = set()
        for component in self.components:
            states |= component.states
        return frozenset(states)

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def num_components(self) -> int:
        return len(self.components)

    @cached_property
    def parameters(self) -> frozenset[atoms.Parameter]:
        parameters: set[atoms.Parameter] = set()
        for component in self.components:
            parameters |= component.parameters
        return frozenset(parameters)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    @cached_property
    def state_derivatives(self) -> frozenset[atoms.StateDerivative]:
        state_derivatives: set[atoms.StateDerivative] = set()
        for component in self.components:
            state_derivatives |= component.state_derivatives
        return frozenset(state_derivatives)

    @cached_property
    def intermediates(self) -> frozenset[atoms.Intermediate]:
        intermediates: set[atoms.Intermediate] = set()
        for component in self.components:
            intermediates |= component.intermediates
        return frozenset(intermediates)

    @property
    def symbols(self) -> dict[str, sp.Symbol]:
        return self._symbols

    def __getitem__(self, name) -> atoms.Atom:
        return self._lookup[name]

    @cached_property
    def sorted_assignments(self) -> tuple[str, ...]:
        return sort_assignments(
            self.intermediates | self.state_derivatives,
            assignments_only=True,
        )
