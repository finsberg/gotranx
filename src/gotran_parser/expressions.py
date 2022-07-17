from __future__ import annotations

import lark
import sympy as sp


def build_expression(root: lark.Tree, symbols: dict[str, sp.Symbol]) -> sp.Expr:
    def expr2symbols(tree: lark.Tree):
        if tree.data == "name":
            return symbols[str(tree.children[0])]
        if tree.data == "number":
            return float(tree.children[0])

        first, second = tree.children
        if tree.data == "add":
            return expr2symbols(first) + expr2symbols(second)
        if tree.data == "sub":
            return expr2symbols(first) - expr2symbols(second)
        if tree.data == "div":
            return expr2symbols(first) / expr2symbols(second)
        if tree.data == "mul":
            return expr2symbols(first) * expr2symbols(second)

        # FIXME: Add custom exception
        raise RuntimeError

    return expr2symbols(root)
