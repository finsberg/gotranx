from __future__ import annotations

import enum

from sympy.printing.c import C99CodePrinter
from sympy.printing.cxx import CXX17CodePrinter, _CXXCodePrinterBase

from .. import templates
from ..exceptions import InvalidNamespace
from ..ode import ODE
from .c import CCodeGenerator, Format, GotranCCodePrinter


class FloatType(str, enum.Enum):
    double = "double"
    float = "float"


class GotranCppCodePrinter(_CXXCodePrinterBase, GotranCCodePrinter):
    """The C printer with C++17's `std::` math functions.

    With `float_type="float"` every number is printed as a float literal (`2.5f`,
    `2.0f`, `1.0f/3.0f`), so that no expression is silently promoted to double: in C++,
    `std::pow(x, 2)` and `std::fmax(0, x)` are double for a float `x`.
    """

    printmethod = "_cxxcode"
    language = "C++"
    standard = "C++17"
    reserved_words = CXX17CodePrinter.reserved_words
    _kf = CXX17CodePrinter._kf
    # The C++ base prints std::max/std::min, which do not compile for mixed int and
    # floating point arguments (std::max(0, V)), so keep C's fmax/fmin
    _print_Max = C99CodePrinter._print_Max
    _print_Min = C99CodePrinter._print_Min

    def __init__(self, float_type: str = "double", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.float_type = float_type

    @property
    def _single_precision(self) -> bool:
        return self.float_type == "float"

    def _print_Float(self, flt):
        if self._single_precision:
            return f"{float(flt)!r}f"
        return super()._print_Float(flt)

    def _print_Integer(self, expr):
        if self._single_precision:
            return f"{int(expr)}.0f"
        return super()._print_Integer(expr)

    def _print_Rational(self, expr):
        if self._single_precision:
            return f"{float(expr.p)!r}f/{float(expr.q)!r}f"
        return super()._print_Rational(expr)

    def _print_Indexed(self, expr):
        if self._single_precision:
            # An index is an integer, not a float literal
            index = ", ".join(str(int(i)) for i in expr.indices)
            return f"{self._print(expr.base.label)}[{index}]"
        return super()._print_Indexed(expr)


def valid_namespace(name: str) -> bool:
    return name.isascii() and name.isidentifier() and name not in CXX17CodePrinter.reserved_words


class CppCodeGenerator(CCodeGenerator):
    """C++17 code: the C generator's functions, `inline` and in a namespace.

    `gotran2cpp.get_code` puts the pieces in `namespace <namespace> { ... }`; the
    namespace defaults to the ODE's name.
    """

    def __init__(
        self,
        ode: ODE,
        format: Format = Format.clang_format,
        remove_unused: bool = False,
        namespace: str | None = None,
        float_type: FloatType | str = FloatType.double,
    ) -> None:
        # Before CCodeGenerator.__init__, which derives the type strings from it
        self.float_type = FloatType(float_type).value
        super().__init__(ode, format=format, remove_unused=remove_unused)
        self._printer = GotranCppCodePrinter(float_type=self.float_type)
        name = namespace or ode.name
        if not valid_namespace(name):
            raise InvalidNamespace(name)
        self.namespace = name

    @property
    def template(self):
        return templates.cpp

    def imports(self) -> str:
        return self._format("\n".join(["#pragma once", "#include <cmath>", "#include <cstring>\n"]))

    def constants(self) -> str:
        num_monitored = len(self.ode.state_derivatives) + len(self.ode.intermediates)
        return "\n".join(
            [
                f"constexpr int NUM_STATES = {len(self.ode.states)};",
                f"constexpr int NUM_PARAMS = {len(self.ode.parameters)};",
                f"constexpr int NUM_MONITORED = {num_monitored};",
            ]
        )
