from __future__ import annotations

from .cpp import GotranCppCodePrinter
from .gpu import GPUIndexingMixin


class CUDACodePrinter(GPUIndexingMixin, GotranCppCodePrinter):
    """The C++ printer for CUDA device code.

    Math functions are printed without `std::` (`exp(x)`), which resolves to CUDA's device
    overloads for both float and double. Every integer is printed as a floating point
    literal: `sqrt(5)` would resolve to the C++ library's constexpr integer overload, which
    is host-only and cannot be called from a kernel.
    """

    _ns = ""

    def _print_Integer(self, expr):
        if self._single_precision:
            return super()._print_Integer(expr)
        return f"{int(expr)}.0"
