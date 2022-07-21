from __future__ import annotations

from typing import Optional

import lark
import sympy as sp


def build_expression(
    root: lark.Tree,
    symbols: Optional[dict[str, sp.Symbol]] = None,
) -> sp.Expr:

    symbols_: dict[str, sp.Symbol] = symbols or {}

    def expr2symbols(tree: lark.Tree):
        if tree.data == "variable":
            return symbols_[str(tree.children[0])]
        if tree.data == "scientific":
            return float(tree.children[0])
        if tree.data == "add":
            return expr2symbols(tree.children[0]) + expr2symbols(tree.children[1])
        if tree.data == "sub":
            return expr2symbols(tree.children[0]) - expr2symbols(tree.children[1])
        if tree.data == "div":
            return expr2symbols(tree.children[0]) / expr2symbols(tree.children[1])
        if tree.data == "mul":
            return expr2symbols(tree.children[0]) * expr2symbols(tree.children[1])
        if tree.data == "pow":
            return pow(expr2symbols(tree.children[0]), expr2symbols(tree.children[1]))

        if tree.data == "constant":
            if tree.children[0] == "pi":
                return sp.pi

        if tree.data == "func":
            if tree.children[0] == "log":
                return sp.log(expr2symbols(tree.children[1]))
            if tree.children[0] == "exp":
                return sp.exp(expr2symbols(tree.children[1]))

        # FIXME: Add custom exception
        raise RuntimeError

    return expr2symbols(root)
