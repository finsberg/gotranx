from __future__ import annotations

from .. import templates
from ..ode import ODE
from .c import Format
from .cpp import CppCodeGenerator, FloatType, GotranCppCodePrinter
from .gpu import GPUIndexingMixin, GPUKernelMixin


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


class CUDACodeGenerator(GPUKernelMixin, CppCodeGenerator):
    """CUDA kernels: one thread per cell, all cells' states stored state-major.

    State `i` of cell `cell` is `states[i * n_cells + cell]`, and likewise for the values a
    kernel writes and for the missing variables; the parameters are shared by every cell.
    Each kernel takes the number of cells as its last argument. `states` and `values` are
    not `__restrict__`, so a scheme may update the states in place.
    """

    restrict = "__restrict__"
    backend_name = "CUDA"

    def __init__(
        self,
        ode: ODE,
        format: Format = Format.clang_format,
        remove_unused: bool = False,
        namespace: str | None = None,
        float_type: FloatType | str = FloatType.double,
    ) -> None:
        super().__init__(
            ode,
            format=format,
            remove_unused=remove_unused,
            namespace=namespace,
            float_type=float_type,
        )
        self._printer = CUDACodePrinter(float_type=self.float_type)
        self._check_reserved_names()

    @property
    def template(self):
        return templates.cuda

    def _pointer(self, name: str, const: bool = True) -> str:
        if name == "states":
            # Not __restrict__: the states may be the values a scheme writes (in place)
            prefix = "const " if const else ""
            return f"{prefix}{self.float_type} *{name}"
        return super()._pointer(name, const=const)

    def imports(self) -> str:
        return self._format(
            "\n".join(
                ["#pragma once", "#include <cmath>", "#include <cstdint>", "#include <cstring>\n"]
            )
        )
