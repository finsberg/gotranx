"""What every GPU backend shares, whatever its vendor: one thread per cell, with the states
of all cells stored state-major (state `i` of cell `c` is `states[i * n_cells + c]`)."""

from __future__ import annotations

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
