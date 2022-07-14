from __future__ import annotations

from typing import Iterable
from typing import Sequence
from typing import TypeVar

import attr

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


def find_lhs_symbols(components: Sequence[Component]):
    symbols = []
    for component in components:
        for p in component.parameters:
            symbols.append(p.name)
        for s in component.states:
            symbols.append(s.name)
        for a in component.assignments:
            symbols.append(a.lhs)
    return symbols


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
    symbols: frozenset[str] = attr.ib(init=False)

    def __attrs_post_init__(self):
        check_components(components=self.components)
        symbols = find_lhs_symbols(components=self.components)

        if not len(symbols) == len(set(symbols)):
            raise exceptions.DuplicateSymbolError(find_duplicates(symbols))
