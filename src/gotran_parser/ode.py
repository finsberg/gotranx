from __future__ import annotations

import re

from attrs import define
from attrs import field

from . import atoms
from . import exceptions

_STATE_DERIV_EXPR = re.compile(r"d([A-Z,a-z])\w*_dt")


@define(frozen=True)
class Component:
    name: str
    states: frozenset[atoms.State]
    parameters: frozenset[atoms.Parameter]
    assignments: frozenset[atoms.Assignment]
    state_derivatives: frozenset[atoms.StateDerivative] = field(init=False)
    intermediates: frozenset[atoms.Intermediate] = field(init=False)

    def __attrs_post_init__(self):
        self._handle_assignments()

    def _find_state(self, state_name: str) -> atoms.State:
        for state in self.states:
            if state.name == state_name:
                return state
        else:
            raise exceptions.StateNotFound(state_name=state_name)

    def _handle_assignments(self):
        state_derivatives = []
        intermediates = []
        for assignment in self.assignments:
            if state_name := _STATE_DERIV_EXPR.match(assignment.lhs):
                state = self._find_state(state_name=state_name.group(1))
                state_derivatives.append(assignment.to_state_derivative(state))
            else:
                intermediates.append(assignment.to_intermediate())

        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        object.__setattr__(self, "intermediates", set(intermediates))
        object.__setattr__(self, "state_derivatives", set(state_derivatives))

    def is_complete(self):
        """Check that all states have a corresponding state derivative"""
        states = set()
        for state_derivative in self.state_derivatives:
            states.add(state_derivative.state)
        return states == self.states
