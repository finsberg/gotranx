from __future__ import annotations
import functools
import sympy
from sympy.codegen.ast import Assignment

from .python import PythonCodeGenerator, GotranPythonCodePrinter, _print_Piecewise
from .. import templates


class UFLPrinter(GotranPythonCodePrinter):
    _kf = {
        "exp": "ufl.exp",
        "log": "ufl.ln",
        "sin": "ufl.sin",
        "cos": "ufl.cos",
        "tan": "ufl.tan",
        "asin": "ufl.asin",
        "acos": "ufl.acos",
        "atan": "ufl.atan",
        "sinh": "ufl.sinh",
        "cosh": "ufl.cosh",
        "tanh": "ufl.tanh",
        "sqrt": "ufl.sqrt",
        "Abs": "abs",  # SymPy's Abs maps to Python's built-in abs, which UFL intercepts
    }

    def _hprint_Pow(self, expr, rational=False, sqrt="ufl.sqrt"):
        return super(GotranPythonCodePrinter, self)._hprint_Pow(expr, rational, sqrt)

    def _print_Piecewise(self, expr):
        result = []

        if isinstance(expr.args[0][0], Assignment):
            lhs = super(GotranPythonCodePrinter, self)._print(expr.args[0][0].lhs)
            result.append(f"{lhs} = ")
            all_lsh_equal = True
            for arg in expr.args:
                result.append("ufl.conditional(")
                result.append(f"{super(GotranPythonCodePrinter, self)._print(arg[1])}")
                result.append(", ")
                result.append(f"{super(GotranPythonCodePrinter, self)._print(arg[0].rhs)}")
                result.append(", ")
                all_lsh_equal = (
                    all_lsh_equal and super(GotranPythonCodePrinter, self)._print(arg[0].lhs) == lhs
                )

            assert all_lsh_equal, "All assignments in Piecewise must have the same lhs"

            if super(GotranPythonCodePrinter, self)._print(arg[1]) == "True":
                result = result[:-6]
                result.append(f", {super(GotranPythonCodePrinter, self)._print(arg[0].rhs)}")
            else:
                raise ValueError("Last condition in Piecewise must be True")

            result.append(")" * (len(expr.args) - 1))

        else:
            conds, exprs = _print_Piecewise(self, expr)

            for c, e in zip(conds, exprs):
                result.append("ufl.conditional(")
                result.append(f"{c}")
                result.append(", ")
                result.append(f"{e}")
                result.append(", ")

            if c == "True":
                result = result[:-6]
                result.append(f", {e}")
            else:
                raise ValueError("Last condition in Piecewise must be True")

            result.append(")" * (len(conds) - 1))

        return "".join(result)

    def _print_And(self, expr):
        args = [self._print(arg) for arg in expr.args]
        return functools.reduce(lambda x, y: f"ufl.And({x}, {y})", args)

    def _print_Or(self, expr):
        args = [self._print(arg) for arg in expr.args]
        return functools.reduce(lambda x, y: f"ufl.Or({x}, {y})", args)

    def _print_Min(self, expr):
        # ufl.min_value only takes two arguments, so reduce multi-argument Mins
        args = [self._print(arg) for arg in expr.args]
        return functools.reduce(lambda x, y: f"ufl.min_value({x}, {y})", args)

    def _print_Max(self, expr):
        # ufl.max_value only takes two arguments, so reduce multi-argument Max
        args = [self._print(arg) for arg in expr.args]
        return functools.reduce(lambda x, y: f"ufl.max_value({x}, {y})", args)

    def _print_sign(self, e):
        return f"ufl.sign({self._print(e.args[0])})"

    def _print_Assignment(self, expr):
        sym, value = expr.lhs, expr.rhs
        if isinstance(sym, sympy.tensor.indexed.Indexed):
            if sym.base.name == "values":
                index = self._print(sym.indices[0])
                return f"_{sym.base.name}_{index} = {self._print(value)}"
        return super()._print_Assignment(expr)


class UFLCodeGenerator(PythonCodeGenerator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._printer = UFLPrinter()

    def imports(self) -> str:
        return self._format("import ufl\nimport numpy\n")

    @property
    def template(self):
        return templates.ufl
