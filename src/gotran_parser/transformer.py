from __future__ import annotations

from collections import defaultdict

import lark

from . import atoms


class TreeToODE(lark.Transformer):
    def states(self, s) -> list[atoms.State]:
        return [
            atoms.State(name=str(p[0]), ic=float(p[1]), component=s[0], info=s[1])
            for p in s[2:]
        ]

    def parameters(self, s) -> list[atoms.Parameter]:
        return [
            atoms.Parameter(name=str(p[0]), value=float(p[1]), component=s[0])
            for p in s[1:]
        ]

    def pair(self, s):
        name, value = s
        return (name, value)

    def assignment(self, s) -> list[atoms.Assignment]:
        return [atoms.Assignment(lhs=str(s[0]), rhs=atoms.Expression(s[1]))]

    def ode(self, s):
        mapping = {
            atoms.Parameter: "parameters",
            atoms.Assignment: "assignments",
            atoms.State: "states",
        }
        components = defaultdict(
            lambda: {"assignments": set(), "parameters": set(), "states": set()},
        )

        for line in s:
            for value in line:
                components[value.component][mapping[type(value)]].add(value)

        return [atoms.Component(name=key, **value) for key, value in components.items()]
