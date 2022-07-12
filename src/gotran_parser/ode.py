from dataclasses import dataclass

from . import atoms


@dataclass(frozen=True)
class Component:
    name: str
    states: frozenset[atoms.State]
    parameters: frozenset[atoms.Parameter]
    assignments: frozenset[atoms.Assignment]

    @property
    def intermediates(self):
        pass

    @property
    def state_derivatives(self):
        pass
