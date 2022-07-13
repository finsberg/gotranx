from __future__ import annotations

from typing import Optional
from typing import Union

import attr
import lark


@attr.s(frozen=True, kw_only=True, slots=True)
class Parameter:
    name: str = attr.ib()
    value: float = attr.ib()
    component: Optional[str] = attr.ib(None)


@attr.s(frozen=True, kw_only=True, slots=True)
class State:
    name: str = attr.ib()
    ic: float = attr.ib()
    component: Optional[str] = attr.ib(None)
    info: Optional[str] = attr.ib(None)


@attr.s(frozen=True, kw_only=True, slots=True)
class Expression:
    tree: lark.Tree = attr.ib()
    dependencies: frozenset[str] = attr.ib(init=False)

    def __attrs_post_init__(self):
        self._find_dependencies()

    def _find_dependencies(self):
        deps = set()
        for tree in self.tree.iter_subtrees():
            if tree.data == "name":
                deps.add(str(tree.children[0]))

        object.__setattr__(self, "dependencies", frozenset(deps))


@attr.s(frozen=True, kw_only=True, slots=True)
class Assignment:
    lhs: str = attr.ib()
    rhs: Expression = attr.ib()
    component: Optional[str] = attr.ib(None)

    def to_intermediate(self) -> "Intermediate":
        return Intermediate(lhs=self.lhs, rhs=self.rhs, component=self.component)

    def to_state_derivative(self, state: State) -> "StateDerivative":
        return StateDerivative(
            lhs=self.lhs,
            rhs=self.rhs,
            component=self.component,
            state=state,
        )


@attr.s(frozen=True, kw_only=True, slots=True)
class Intermediate:
    lhs: str = attr.ib()
    rhs: Expression = attr.ib()
    component: Optional[str] = attr.ib(None)


@attr.s(frozen=True, kw_only=True, slots=True)
class StateDerivative:
    """Derivative of a state."""

    lhs: str = attr.ib()
    rhs: Expression = attr.ib()
    state: State = attr.ib()
    component: Optional[str] = attr.ib(None)

    def __attr_post_init__(self):
        # Check that state derivative and state
        # has the same component
        pass


Atoms = Union[State, Parameter, Assignment]
