from __future__ import annotations

import re

import attr

from . import atoms
from . import exceptions


STATE_DERIV_EXPR = re.compile(r"^d(?P<state>\w+)_dt$")


@attr.s(frozen=True, slots=True)
class BaseComponent:
    """A Component is a collection of states, parameters and
    assignments that are given the same tag. This is useful if
    you have a complicated set of equations and need to group them together.

    Raises
    ------
    StateNotFound
        If component contains a state derivative but not the State
    """

    name: str = attr.ib()
    states: frozenset[atoms.State] = attr.ib()
    parameters: frozenset[atoms.Parameter] = attr.ib()
    assignments: frozenset[atoms.Assignment] = attr.ib(init=False)
    state_derivatives: frozenset[atoms.StateDerivative] = attr.ib(init=False)
    intermediates: frozenset[atoms.Intermediate] = attr.ib(init=False)

    def simplify(self) -> Component:
        """Run sympy's simplify function on all expressions in the ODE"""
        return Component(
            name=self.name,
            states=self.states,
            parameters=self.parameters,
            assignments=frozenset([assignment.simplify() for assignment in self.assignments]),
        )

    def find_state(self, state_name: str) -> atoms.State:
        """Find a state by name

        Parameters
        ----------
        state_name : str
            The name of the state

        Returns
        -------
        atoms.State
            The state

        Raises
        ------
        exceptions.StateNotFoundInComponent
            If state is not found in component
        """
        for state in self.states:
            if state.name == state_name:
                return state
        else:
            raise exceptions.StateNotFoundInComponent(
                state_name=state_name,
                component_name=self.name,
            )

    def find_parameter(self, parameter_name: str) -> atoms.Parameter:
        """Find a parameter by name

        Parameters
        ----------
        parameter_name : str
            The name of the parameter

        Returns
        -------
        atoms.Parameter
            The parameter

        Raises
        ------
        exceptions.ParameterNotFoundInComponent
            If parameter is not found in component
        """

        for parameter in self.parameters:
            if parameter.name == parameter_name:
                return parameter
        else:
            raise exceptions.ParameterNotFoundInComponent(
                parameter_name=parameter_name,
                component_name=self.name,
            )

    def find_assignment(self, assignment_name: str) -> atoms.Assignment:
        """Find a assignment by name

        Parameters
        ----------
        assignment_name : str
            The name of the assignment

        Returns
        -------
        atoms.Assignment
            The assignment

        Raises
        ------
        exceptions.AssignmentNotFoundInComponent
            If assignment is not found in component
        """

        for assignment in self.assignments:
            if assignment.name == assignment_name:
                return assignment
        else:
            raise exceptions.AssignmentNotFoundInComponent(
                assignment_name=assignment_name,
                component_name=self.name,
            )

    def is_complete(self) -> bool:
        """Returns true if all states have a corresponding state derivative"""

        return self.states_with_derivatives == self.states

    @property
    def states_with_derivatives(self) -> frozenset[atoms.State]:
        """Returns the states that have a corresponding state derivative

        Returns
        -------
        frozenset[atoms.State]
            The states that have a corresponding state derivative
        """
        states = set()
        for state_derivative in self.state_derivatives:
            states.add(state_derivative.state)
        return frozenset(states)

    @property
    def states_without_derivatives(self) -> frozenset[atoms.State]:
        """Returns the states that do not have a corresponding state derivative

        Returns
        -------
        frozenset[atoms.State]
            The states that do not have a corresponding state derivative
        """
        return self.states.difference(self.states_with_derivatives)

    @property
    def atoms(self) -> frozenset[atoms.Atom]:
        """Returns all atoms in the component

        Returns
        -------
        frozenset[atoms.Atom]
            All atoms in the component
        """
        return frozenset(
            self.states.union(self.parameters).union(self.assignments).union(self.intermediates)
        )

    def to_ode(self):
        from .ode import ODE

        """Convert component to ODE"""
        return ODE(
            components=(self,),
            name=self.name,
        )


@attr.s(frozen=True, slots=True)
class Component(BaseComponent):
    """A Component is a collection of states, parameters and
    assignments that are given the same tag. This is useful if
    you have a complicated set of equations and need to group them together.

    Raises
    ------
    StateNotFound
        If component contains a state derivative but not the State
    """

    assignments: frozenset[atoms.Assignment] = attr.ib()

    def __attrs_post_init__(self):
        self._handle_assignments()

    def remove_singularities(self, lookup: dict[str, atoms.Atom]) -> Component:
        """Remove singularities from the component

        Parameters
        ----------
        lookup : dict[str, atoms.Atom]
            A lookup table for atoms

        Returns
        -------
        Component
            A new component with singularities removed
        """
        new_assignments = set()

        for assignment in self.assignments:
            new_assignments.add(assignment.remove_singularities(lookup))

        return Component(
            name=self.name,
            states=self.states,
            parameters=self.parameters,
            assignments=frozenset(new_assignments),
        )

    def _handle_assignments(self):
        """Handle assignments and create intermediates and state derivatives"""
        state_derivatives = set()
        intermediates = set()
        for assignment in self.assignments:
            if isinstance(assignment, atoms.Intermediate):
                intermediates.add(assignment)
            elif isinstance(assignment, atoms.StateDerivative):
                state_derivatives.add(assignment)
            else:
                if state_name := STATE_DERIV_EXPR.match(assignment.name):
                    state = self.find_state(state_name=state_name.groupdict()["state"])
                    state_derivatives.add(assignment.to_state_derivative(state))
                else:
                    intermediates.add(assignment.to_intermediate())

        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        object.__setattr__(self, "intermediates", intermediates)
        object.__setattr__(self, "state_derivatives", state_derivatives)


@attr.s(frozen=True, slots=True)
class MyokitComponent(BaseComponent):
    """A Component is a collection of states, parameters and
    assignments that are given the same tag. This is useful if
    you have a complicated set of equations and need to group them together.

    Raises
    ------
    StateNotFound
        If component contains a state derivative but not the State
    """

    state_derivatives: frozenset[atoms.StateDerivative] = attr.ib()
    intermediates: frozenset[atoms.Intermediate] = attr.ib()

    def __attrs_post_init__(self):
        object.__setattr__(
            self,
            "assignments",
            frozenset(tuple(self.intermediates) + tuple(self.state_derivatives)),
        )
