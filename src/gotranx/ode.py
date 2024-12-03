from __future__ import annotations
from collections import defaultdict

from functools import cached_property
from pathlib import Path
from graphlib import TopologicalSorter
from collections.abc import Iterable
from collections.abc import Sequence
from typing import TypeVar
from typing import cast
from typing import Any
from typing import NamedTuple

import sympy as sp

from . import atoms
from . import exceptions
from .ode_component import BaseComponent, Component

T = TypeVar("T")
U = TypeVar("U", bound=atoms.Assignment)


def check_components(components: Sequence[BaseComponent]):
    """Check if all components are complete

    Parameters
    ----------
    components : Sequence[gotranx.ode_component.BaseComponent]
        The components to check

    Raises
    ------
    exceptions.ComponentNotCompleteError
        If a component is not complete
    """
    for comp in components:
        if not comp.is_complete():
            raise exceptions.ComponentNotCompleteError(
                component_name=comp.name,
                missing_state_derivatives=[state.name for state in comp.states_without_derivatives],
            )


class AllAtoms(NamedTuple):
    symbol_names: list[str]
    symbol_values: dict[str, set[Any]]
    symbols: dict[str, sp.Symbol]
    lookup: dict[str, atoms.Atom]


def gather_atoms(
    components: Sequence[BaseComponent],
) -> AllAtoms:
    """Gather all atoms from a list of components

    Parameters
    ----------
    components : Sequence[gotranx.ode_component.BaseComponent]
        The components to gather atoms from

    Returns
    -------
    AllAtoms
        A named tuple containing all atoms

    """
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
    return AllAtoms(symbol_names, symbol_values, symbols, lookup)


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
    components: Sequence[BaseComponent],
    t: sp.Symbol,
) -> tuple[Component, ...]:
    """Add a temporal state to all components

    Parameters
    ----------
    components : Sequence[gotranx.ode_component.BaseComponent]
        The components to add the temporal state to
    t : sp.Symbol
        The temporal symbol

    Returns
    -------
    tuple[gotranx.ode_component.Component, ...]
        The new components with the temporal state
    """
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
    """Resolve all expressions in a list of components

    Parameters
    ----------
    components : Sequence[gotranx.ode_component.Component]
        The components to resolve expressions in
    symbols : dict[str, sp.Symbol]
        The symbols to resolve the expressions with

    Returns
    -------
    tuple[gotranx.ode_component.Component, ...]
        The new components with resolved expressions
    """
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
    """Create an ODE from a list of components

    Parameters
    ----------
    components : Sequence[gotranx.ode_component.Component]
        The components to create the ODE from
    comments : Sequence[atoms.Comment] | None, optional
        The a list of comments, by default None
    name : str, optional
        Name of the ODE, by default "ODE"

    Returns
    -------
    ODE
        The ODE

    Raises
    ------
    exceptions.DuplicateSymbolError
        If a symbol is duplicated
    """
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
    """Sort assignments by dependencies using a topological sorter

    Parameters
    ----------
    assignments : Iterable[atoms.Assignment]
        The assignments to sort
    assignments_only : bool, optional
        If True only include assignments. If False you also include
        states and parameters, by default True.

    Returns
    -------
    tuple[str, ...]
        The sorted assignments

    Raises
    ------
    exceptions.GotranxError
        If an assignment has a None value
    """
    sorter: TopologicalSorter = TopologicalSorter()
    assignment_names = set()
    for assignment in assignments:
        assignment_names.add(assignment.name)
        if assignment.value is None:
            msg = (
                "Unable to sort assignments with None value"
                "Try to save the ODE to an .ode file first and load it again"
            )
            raise exceptions.GotranxError(msg)
        sorter.add(assignment.name, *assignment.value.dependencies)

    static_order = tuple(sorter.static_order())

    if assignments_only:
        static_order = tuple([name for name in static_order if name in assignment_names])

    return static_order


class ODE:
    """A class representing an ODE

    Parameters
    ----------
    components : Sequence[gotranx.ode_component.BaseComponent]
        The components of the ODE
    t : sp.Symbol | None, optional
        Symbol representing time, by default None
    name : str, optional
        Name of the ODE, by default "ODE"
    comments : Sequence[atoms.Comment] | None, optional
        List of comments, by default None

    Raises
    ------
    exceptions.DuplicateSymbolError
        If a symbol is duplicated

    """

    def __init__(
        self,
        components: Sequence[BaseComponent],
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
        self._components = {comp.name: comp for comp in components}

        self.comments = comments
        self.text = " ".join(comment.text for comment in comments)

    def remove_singularities(self):
        new_components: list[BaseComponent] = []
        for component in self._components.values():
            new_components.append(component.remove_singularities(self._lookup))

        return ODE(
            components=new_components,
            t=self.t,
            name=self.name,
            comments=self.comments,
        )

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
        """Get all states in the ODE"""
        states: set[atoms.State] = set()
        for component in self.components:
            states |= component.states
        return tuple(sorted(states, key=lambda x: x.name))

    @property
    def num_states(self) -> int:
        """Get the number of states in the ODE"""
        return len(self.states)

    @property
    def num_components(self) -> int:
        """Get the number of components in the ODE"""
        return len(self.components)

    @cached_property
    def parameters(self) -> tuple[atoms.Parameter, ...]:
        """Get all parameters in the ODE"""
        parameters: set[atoms.Parameter] = set()
        for component in self.components:
            parameters |= component.parameters
        return tuple(sorted(parameters, key=lambda x: x.name))

    @property
    def num_parameters(self) -> int:
        """Get the number of parameters in the ODE"""
        return len(self.parameters)

    @cached_property
    def state_derivatives(self) -> tuple[atoms.StateDerivative, ...]:
        """Get all state derivatives in the ODE sorted by name"""
        state_derivatives: set[atoms.StateDerivative] = set()
        for component in self.components:
            state_derivatives |= component.state_derivatives
        return tuple(sorted(state_derivatives, key=lambda x: x.name))

    @cached_property
    def intermediates(self) -> tuple[atoms.Intermediate, ...]:
        """Get all intermediates in the ODE sorted by name"""
        intermediates: set[atoms.Intermediate] = set()
        for component in self.components:
            intermediates |= component.intermediates
        return tuple(sorted(intermediates, key=lambda x: x.name))

    @property
    def symbols(self) -> dict[str, sp.Symbol]:
        """Get a dictionary of all symbols in the ODE
        with the symbol name as key and the symbol as value
        """
        return self._symbols

    def __sub__(self, other: BaseComponent) -> ODE:
        new_components = [comp for comp in self.components if comp != other]
        return ODE(
            components=new_components,
            t=self.t,
            name=f"{self.name} - {other.name}",
            comments=self.comments,
        )

    def __getitem__(self, name) -> atoms.Atom:
        return self._lookup[name]

    def get_component(self, name: str) -> BaseComponent:
        """Get a component by name

        Parameters
        ----------
        name : str
            Name of the component
        """
        return self._components[name]

    def sorted_assignments(
        self, assignments_only: bool = True, remove_unused: bool = False
    ) -> tuple[atoms.Assignment, ...]:
        """Get the assignments in the ODE sorted by dependencies

        Parameters
        ----------
        assignments_only : bool, optional
            If True only return assignments otherwise you can
            include states and parameters as well, by default True
        remove_unused : bool, optional
            Remove unused variables, by default False

        Returns
        -------
        tuple[atoms.Assignment, ...]
            The sorted assignments
        """
        intermediates = self.intermediates
        if remove_unused:
            deps = self.dependents()
            intermediates = tuple([a for a in intermediates if a.name in deps])

        names = sort_assignments(
            assignments=intermediates + self.state_derivatives,
            assignments_only=assignments_only,
        )
        return tuple([cast(atoms.Assignment, self[name]) for name in names])

    def sorted_state_derivatives(self) -> tuple[atoms.StateDerivative, ...]:
        """Get the state derivatives in the ODE sorted by dependencies"""
        return tuple(s for s in self.sorted_assignments() if isinstance(s, atoms.StateDerivative))

    def sorted_states(self) -> tuple[atoms.State, ...]:
        """Get the states in the ODE sorted by dependencies"""
        return tuple(s.state for s in self.sorted_state_derivatives())

    def save(self, path: Path) -> None:
        """Save the ODE to a file

        Parameters
        ----------
        path : Path
            The path to save the ODE to
        """
        from .save import write_ODE_to_ode_file

        write_ODE_to_ode_file(self, path)

    def dependents(self) -> dict[str, set[str]]:
        """Get a dictionary of dependents for each component"""
        dependencies = defaultdict(set)
        for component in self.components:
            for assignment in component.assignments:
                if assignment.value is None:
                    msg = (
                        "Unable to sort assignments with None value"
                        "Try to save the ODE to an .ode file first and load it again"
                    )
                    raise exceptions.GotranxError(msg)
                for dependency in assignment.value.dependencies:
                    dependencies[dependency].add(assignment.name)

        return dict(dependencies)

    @property
    def missing_variables(self) -> dict[str, int]:
        """Get a dictionary of missing variables for each component

        This is relevant if you have different sub odes where the
        states in one sub ode is a parameter in another sub ode

        """
        symbols = set(self.symbols.keys()) | {"t"}
        variable_names = {var for var in self.dependents() if var not in symbols}
        return {var: i for i, var in enumerate(sorted(variable_names))}
