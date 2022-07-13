from dataclasses import dataclass
from functools import cached_property
from typing import Optional
from typing import Union

import lark


class default_factory:
    def __init__(self, name):
        self.name = name
        self.value = 0

    def __call__(self):
        self.value += 1
        return f"{self.name} {self.value}"


@dataclass(frozen=True)
class Parameter:
    name: str
    value: float
    component: Optional[str] = None


@dataclass(frozen=True)
class State:
    name: str
    ic: float
    component: Optional[str] = None
    info: Optional[str] = None


@dataclass(frozen=True)
class Expression:
    tree: lark.Tree

    @cached_property
    def dependencies(self):
        deps = set()
        for tree in self.tree.iter_subtrees():
            if tree.data == "name":
                deps.add(str(tree.children[0]))
        return deps


@dataclass(frozen=True, kw_only=True)
class Assignment:
    lhs: str
    rhs: Expression
    component: Optional[str] = None

    def to_intermediate(self) -> "Intermediate":
        return Intermediate(lhs=self.lhs, rhs=self.rhs, component=self.component)

    def to_state_derivative(self, state: State) -> "StateDerivative":
        return StateDerivative(
            lhs=self.lhs,
            rhs=self.rhs,
            component=self.component,
            state=state,
        )


@dataclass(frozen=True)
class Intermediate(Assignment):
    pass


@dataclass(frozen=True, kw_only=True)
class StateDerivative(Assignment):
    """Derivative of a state.

    All StateDerivatives are also Assignments.

    """

    state: State

    def __post_init__(self):
        # Check that state derivative and state
        # has the same component
        pass


Atoms = Union[State, Parameter, Assignment]
