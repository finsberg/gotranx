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
class Assignment:
    lhs: str
    rhs: lark.Tree


class TreeToODE(lark.Transformer):
    def states(self, s):
        return [State(name=str(p[0]), ic=float(p[1])) for p in s]

    def parameters(self, s) -> list[Parameter]:
        return [Parameter(name=str(p[0]), value=float(p[1])) for p in s]

    def expression(self, s):
        return "".join(s)

    def pair(self, s):
        name, value = s
        return (name, value)

    def assignment(self, s):
        return Assignment(lhs=str(s[0]), rhs=s[1])
