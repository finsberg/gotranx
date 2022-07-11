from __future__ import annotations

import lark

from . import atoms


class TreeToODE(lark.Transformer):
    def states(self, s):
        return [atoms.State(name=str(p[0]), ic=float(p[1])) for p in s]

    def parameters(self, s) -> list[atoms.Parameter]:
        return [atoms.Parameter(name=str(p[0]), value=float(p[1])) for p in s]

    def pair(self, s):
        name, value = s
        return (name, value)

    def assignment(self, s) -> atoms.Assignment:
        return atoms.Assignment(lhs=str(s[0]), rhs=atoms.Expression(s[1]))
