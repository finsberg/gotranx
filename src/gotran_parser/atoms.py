from dataclasses import dataclass

import lark


@dataclass(frozen=True)
class Parameter:
    name: str
    value: float


@dataclass(frozen=True)
class State:
    name: str
    ic: float


@dataclass(frozen=True)
class Expression:
    tree: lark.Tree

    @property
    def dependencies(self):
        pass


@dataclass(frozen=True)
class Assignment:
    lhs: str
    rhs: Expression
