from __future__ import annotations

import lark
import sympy as sp
from . import sympytools
from .exceptions import InvalidTreeError, MissingSymbolError


def relational_to_piecewise(expr: sp.Expr) -> sp.Piecewise:
    """Convert a relational expression to a piecewise expression

    Parameters
    ----------
    expr : sp.Expr
        The expression to convert

    Returns
    -------
    sp.Piecewise
        The piecewise expression
    """
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

    if op == "+":
        return sp.Add(fst, snd, evaluate=False)
    if op == "-":
        return sp.Add(fst, sp.Mul(sp.Integer(-1), snd, evaluate=False), evaluate=False)
    if op == "/":
        return sp.Mul(fst, sp.Pow(snd, sp.Integer(-1), evaluate=False), evaluate=False)
    if op == "*":
        return sp.Mul(fst, snd, evaluate=False)
    if op == "**":
        return sp.Pow(fst, snd, evaluate=False)

    raise RuntimeError(f"Invalid binary operation {op}")


def unary_op(op: str, arg):
    """Unary operation

    Parameters
    ----------
    op : str
        Operation to perform
    arg : sp.Expr
        The argument

    Returns
    -------
    sp.Expr
        The result of the operation
    """
    if op == "-":
        return sp.Mul(sp.Integer(-1), arg, evaluate=False)
    if op == "+":
        return arg

    raise RuntimeError(f"Invalid unary operation {op}")


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
        if tree.data in ("expression", "term"):
            fst = expr2symbols(tree.children[0])
            # It is always an odd number of children
            # and at least one
            for i in range(1, len(tree.children), 2):
                fst = binary_op(
                    tree.children[i],
                    fst,
                    expr2symbols(tree.children[i + 1]),
                )
            return fst

        if tree.data == "factor":
            return unary_op(tree.children[0], expr2symbols(tree.children[1]))
        if tree.data == "power":
            return binary_op(
                "**",
                expr2symbols(tree.children[0]),
                expr2symbols(tree.children[1]),
            )

        if tree.data == "variable":
            try:
                return symbols_[str(tree.children[0])]
            except KeyError as e:
                raise MissingSymbolError(
                    symbol=str(tree.children[0]), line_no=tree.meta.line
                ) from e

        if tree.data == "scientific":
            return sp.sympify(tree.children[0])

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

            return getattr(sp, funcname)(*[expr2symbols(c) for c in tree.children[1:]])

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
                *[expr2symbols(c) for c in tree.children[1:]],
            )

        raise InvalidTreeError(tree=tree)

    # Recursively build the expression starting from the root
    return expr2symbols(root)
