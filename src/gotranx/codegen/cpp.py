from __future__ import annotations

from sympy.printing.c import C99CodePrinter
from sympy.printing.cxx import CXX17CodePrinter, _CXXCodePrinterBase

from .c import GotranCCodePrinter


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
