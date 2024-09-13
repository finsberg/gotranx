from __future__ import annotations

import sympy
from .. import templates
from .python import PythonCodeGenerator, GotranPythonCodePrinter


class JaxPrinter(GotranPythonCodePrinter):
    def _print_Assignment(self, expr):
        sym, value = expr.lhs, expr.rhs
        if isinstance(sym, sympy.tensor.indexed.Indexed):
            if sym.base.name == "values":
                index = self._print(sym.indices[0])
                return f"_{sym.base.name}_{index} = {self._print(value)}"

        return super()._print_Assignment(expr)


class JaxCodeGenerator(PythonCodeGenerator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._printer = JaxPrinter()

    def imports(self) -> str:
        return "\n".join(
            [
                "import jax",
                "import jax.numpy as numpy",
                'jax.config.update("jax_enable_x64", True)',
            ]
        )

    @property
    def template(self):
        return templates.jax
