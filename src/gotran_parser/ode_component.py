from __future__ import annotations

import re

import attr

from . import atoms
from . import exceptions


STATE_DERIV_EXPR = re.compile(r"^d(?P<state>\w+)_dt$")


@attr.s(frozen=True, slots=True)
class Component:
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
    assignments: frozenset[atoms.Assignment] = attr.ib()
    state_derivatives: frozenset[atoms.StateDerivative] = attr.ib(init=False)
    intermediates: frozenset[atoms.Intermediate] = attr.ib(init=False)

    def __attrs_post_init__(self):
        self._handle_assignments()

    def _find_state(self, state_name: str) -> atoms.State:
        for state in self.states:
            if state.name == state_name:
                return state
        else:
            raise exceptions.StateNotFoundInComponent(
                state_name=state_name,
                component_name=self.name,
            )

    def _handle_assignments(self):
        state_derivatives = []
        intermediates = []
        for assignment in self.assignments:
            if state_name := STATE_DERIV_EXPR.match(assignment.lhs):
                state = self._find_state(state_name=state_name.groupdict()["state"])
                state_derivatives.append(assignment.to_state_derivative(state))
            else:
                intermediates.append(assignment.to_intermediate())

        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        object.__setattr__(self, "intermediates", set(intermediates))
        object.__setattr__(self, "state_derivatives", set(state_derivatives))

    def is_complete(self) -> bool:
        """Returns true if all states have a corresponding state derivative"""

        return self.states_with_derivatives == self.states

    @property
    def states_with_derivatives(self) -> frozenset[atoms.State]:
        states = set()
        for state_derivative in self.state_derivatives:
            states.add(state_derivative.state)
        return frozenset(states)

    @property
    def states_without_derivatives(self) -> frozenset[atoms.State]:
        return self.states.difference(self.states_with_derivatives)
