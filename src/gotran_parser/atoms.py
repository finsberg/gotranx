from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from typing import Optional

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
class ParameterComponent:
    parameters: list[Parameter]
    component: str = field(init=False)

    def __post_init__(self):
        pass


@dataclass(frozen=True)
class State:
    name: str
    ic: float
    component: Optional[str] = None
    info: Optional[str] = None


@dataclass(frozen=True)
class StateComponent:
    states: list[State]
    component: str = field(init=False)

    def __post_init__(self):
        pass


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


@dataclass(frozen=True)
class Assignment:
    lhs: str
    rhs: Expression
