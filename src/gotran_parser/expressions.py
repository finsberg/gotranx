from __future__ import annotations

import lark
import sympy as sp


def build_expression(root: lark.Tree, symbols: dict[str, sp.Symbol]) -> sp.Expr:
    def expr2symbols(tree: lark.Tree):
        if tree.data == "variable":
            return symbols[str(tree.children[0])]
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

        if tree.data == "func":
            if tree.children[0] == "log":
                return sp.log(expr2symbols(tree.children[1]))

        # FIXME: Add custom exception
        raise RuntimeError

    return expr2symbols(root)
