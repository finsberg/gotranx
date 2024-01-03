from __future__ import annotations

import lark
import sympy as sp
from . import sympytools
from .exceptions import InvalidTreeError

BINARY_OPERATIONS = {"add", "mul", "sub", "div", "pow"}


def relational_to_piecewise(expr: sp.Expr) -> sp.Piecewise:
    if expr.is_Relational:
        return sp.Piecewise(
            (1, expr),
            (0, True),
        )
    return expr


def binary_op(op: str, fst, snd):
    """Binary operation

    Parameters
    ----------
    op : str
        Operation to perform
    fst : sp.Expr
        First argument
    snd : sp.Expr
        Second argument

    Returns
    -------
    sp.Expr
        The result of the operation
    """
    fst = relational_to_piecewise(fst)
    snd = relational_to_piecewise(snd)

    if op == "add":
        return sp.Add(fst, snd, evaluate=False)
    if op == "sub":
        return sp.Add(fst, sp.Mul(sp.Integer(-1), snd, evaluate=False), evaluate=False)
    if op == "div":
        return sp.Mul(fst, sp.Pow(snd, sp.Integer(-1), evaluate=False), evaluate=False)
    if op == "mul":
        return sp.Mul(fst, snd, evaluate=False)
    if op == "pow":
        return sp.Pow(fst, snd, evaluate=False)

    raise RuntimeError(f"Invalid binary operation {op}")


def build_expression(
    root: lark.Tree,
    symbols: dict[str, sp.Symbol] | None = None,
) -> sp.Expr:
    """Build a sympy expression from a lark tree

    Parameters
    ----------
    root : lark.Tree
        The root of the tree
    symbols : dict[str, sp.Symbol], optional
        A dictionary with symbols, by default None

    Returns
    -------
    sp.Expr
        The sympy expression
    """
    symbols_: dict[str, sp.Symbol] = symbols or {}

    def expr2symbols(tree: lark.Tree):
        if tree.data == "variable":
            return symbols_[str(tree.children[0])]
        if tree.data == "scientific":
            return sp.sympify(tree.children[0])

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
                return sympytools.Conditional(
                    cond=expr2symbols(tree.children[1]),
                    true_value=expr2symbols(tree.children[2]),
                    false_value=expr2symbols(tree.children[3]),
                )

            elif tree.children[0] == "ContinuousConditional":
                rel_op, arg1, arg2 = tree.children[1].children
                cond = sp.sympify(rel_op.value)(expr2symbols(arg1), expr2symbols(arg2))

                true_value = expr2symbols(tree.children[2])
                false_value = expr2symbols(tree.children[3])
                sigma = expr2symbols(tree.children[4])

                return sympytools.ContinuousConditional(
                    cond=cond,
                    true_value=true_value,
                    false_value=false_value,
                    sigma=sigma,
                )

            return getattr(sp, tree.children[0])(
                expr2symbols(tree.children[1]),
                expr2symbols(tree.children[2]),
            )

        raise InvalidTreeError(tree=tree)

    return expr2symbols(root)
