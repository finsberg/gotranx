from __future__ import annotations

from typing import Optional

import lark
import sympy as sp

from .exceptions import InvalidTreeError

BINARY_OPERATIONS = {"add", "mul", "sub", "div", "pow"}


def binary_op(op: str, fst, snd):
    if op == "add":
        return fst + snd
    if op == "sub":
        return fst - snd
    if op == "div":
        return fst / snd
    if op == "mul":
        return fst * snd
    if op == "pow":
        return pow(fst, snd)

    raise RuntimeError(f"Invalid binary operation {op}")


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

        if tree.data == "signedatom":
            if tree.children[0] == "-":
                return -expr2symbols(tree.children[1])
            else:  # +
                return expr2symbols(tree.children[1])

        if tree.data in BINARY_OPERATIONS:
            return binary_op(
                tree.data,
                expr2symbols(tree.children[0]),
                expr2symbols(tree.children[1]),
            )

        if tree.data == "constant":
            if tree.children[0] == "pi":
                return sp.pi

        if tree.data == "func":
            # Children name (e.g 'log', 'exp', 'cos' etc) are methods
            # available in the sympy name space
            funcname = tree.children[0]
            if tree.children[0] == "abs":
                # Only exceptions is 'abs' which is 'Abs'
                funcname = "Abs"

            return getattr(sp, funcname)(expr2symbols(tree.children[1]))

        if tree.data == "logicalfunc":
            if tree.children[0] == "Conditional":
                return sp.Piecewise(
                    (expr2symbols(tree.children[2]), expr2symbols(tree.children[1])),
                    (expr2symbols(tree.children[3]), True),
                )

            return getattr(sp, tree.children[0])(
                expr2symbols(tree.children[1]),
                expr2symbols(tree.children[2]),
            )

        raise InvalidTreeError(tree=tree)

    return expr2symbols(root)
