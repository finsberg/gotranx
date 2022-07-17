from __future__ import annotations

from typing import Iterable
from typing import Sequence
from typing import TypeVar

import attr
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


def find_name_symbols(
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


@attr.s
class ODE:
    components: Sequence[Component] = attr.ib()
    t: sp.Symbol = attr.ib("t")

    def __attrs_post_init__(self):
        check_components(components=self.components)
        symbol_names, symbols, lookup = find_name_symbols(components=self.components)

        if not len(symbol_names) == len(set(symbol_names)):
            raise exceptions.DuplicateSymbolError(find_duplicates(symbol_names))
        self._symbols = symbols
        self._lookup = lookup

    @property
    def symbols(self) -> dict[str, sp.Symbol]:
        return self._symbols

    def __getitem__(self, name) -> atoms.Atom:
        return self._lookup[name]

    def resolve_expressions(self) -> ODE:
        components = []
        for component in self.components:
            assignments = []
            for assignment in component.assignments:
                assignments.append(assignment.resolve_expression(self.symbols))
            components.append(
                Component(
                    name=component.name,
                    states=component.states,
                    parameters=component.parameters,
                    assignments=frozenset(assignments),
                ),
            )
        return ODE(components=components, t=self.t)
