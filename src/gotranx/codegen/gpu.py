"""What every GPU backend shares, whatever its vendor: one thread per cell, with the states
of all cells stored state-major (state `i` of cell `c` is `states[i * n_cells + c]`)."""

from __future__ import annotations

import typing

from ..exceptions import ReservedName

#: Arrays with one column per cell. `parameters` is shared by every cell.
STRIDED_ARRAYS = ("states", "values", "missing_variables")


class GPUIndexingMixin:
    """Print the arrays in `STRIDED_ARRAYS` with the state-major index of this thread's cell.

    The index is built as a string rather than printed as a sympy expression: a printer
    that prints integers as floating point literals would otherwise print `2.0*n_cells`.
    """

    def _print_Indexed(self, expr):
        name = str(expr.base.label)
        (index,) = (int(i) for i in expr.indices)
        if name in STRIDED_ARRAYS:
            return f"{name}[{index} * n_cells + cell]"
        return f"{name}[{index}]"


class GPUKernelMixin:
    """Every function of the model as a kernel with one thread per cell.

    Mixed into a code generator. The template receives the kernel qualifier, the argument
    with the number of cells and the prologue that finds this thread's cell, through the
    `_method_kwargs` and `_init_value_kwargs` hooks.
    """

    kernel_qualifier = "__global__"
    index_type = "std::int64_t"
    thread_index_expr = "static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x"
    #: Names the kernels use for their own locals and arguments
    reserved_names: tuple[str, ...] = ("cell", "n_cells")
    backend_name = "GPU"

    def _check_reserved_names(self) -> None:
        ode = self.ode  # type: ignore[attr-defined]
        names = [
            x.name
            for x in [*ode.states, *ode.parameters, *ode.intermediates, *ode.state_derivatives]
        ] + list(ode.missing_variables)
        for name in names:
            if name in self.reserved_names:
                raise ReservedName(name=name, backend=self.backend_name)

    def _kernel_kwargs(self) -> dict[str, typing.Any]:
        prologue = "\n".join(
            [
                f"const {self.index_type} cell = {self.thread_index_expr};",
                "if (cell >= n_cells) return;",
            ]
        )
        return {
            "qualifier": self.kernel_qualifier,
            "prologue": prologue,
            "cell_count_argument": f"const {self.index_type} n_cells",
        }

    def _method_kwargs(self) -> dict[str, typing.Any]:
        return self._kernel_kwargs()

    def _init_value_kwargs(self) -> dict[str, typing.Any]:
        return {**super()._init_value_kwargs(), **self._kernel_kwargs()}  # type: ignore[misc]
