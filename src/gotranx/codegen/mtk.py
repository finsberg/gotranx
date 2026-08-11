from __future__ import annotations

from typing import Iterable

import sympy
import structlog

from .base import CodeGenerator, Func, RHSArgument, SchemeArgument
from .julia import GotranJuliaCodePrinter
from .. import templates
from ..ode import ODE
from .. import atoms

logger = structlog.get_logger()


class MTKCodeGenerator(CodeGenerator):
    def __init__(self, ode: ODE, remove_unused: bool = False) -> None:
        super().__init__(ode, remove_unused=remove_unused)
        self._printer = GotranJuliaCodePrinter()

    @property
    def printer(self):
        return self._printer

    @property
    def template(self):
        return templates.mtk

    # The following methods satisfy the abstract interface but are not used
    def _rhs_arguments(self, order: RHSArgument | str = RHSArgument.tsp, const_states: bool = True):
        dummy = sympy.IndexedBase("dummy")
        return Func(
            arguments=[],
            states=dummy,
            parameters=dummy,
            values=dummy,
            values_type="",
        )

    def _scheme_arguments(
        self,
        order: SchemeArgument | str = SchemeArgument.stdp,
        const_states: bool = True,
    ):
        dummy = sympy.IndexedBase("dummy")
        return Func(
            arguments=[],
            states=dummy,
            parameters=dummy,
            values=dummy,
            values_type="",
        )

    def _format_expr(self, expr) -> str:
        return self.printer.doprint(expr)

    def _observed(self, assignments: Iterable[atoms.Assignment]) -> list[str]:
        observed = []
        for assignment in assignments:
            if isinstance(assignment, atoms.Intermediate):
                observed.append(f"{assignment.name} ~ {self._format_expr(assignment.expr)}")
        return observed

    def _equations(self, derivatives: Iterable[atoms.StateDerivative]) -> list[str]:
        equations = []
        for derivative in derivatives:
            lhs = f"D({derivative.state.name})"
            rhs = self._format_expr(derivative.expr)
            equations.append(f"{lhs} ~ {rhs}")
        return equations

    def generate(self, remove_unused: bool | None = None) -> str:
        """Generate ModelingToolkit.jl code."""
        if remove_unused is None:
            remove_unused = self.remove_unused

        param_names = [p.name for p in self.ode.parameters if self._condition(p.name)]
        missing_names = [n for n in self.ode.missing_variables.keys() if self._condition(n)]
        extra_params = [n for n in missing_names if n not in param_names]

        state_names = [s.name for s in self.ode.sorted_states() if self._condition(s.name)]

        assignments = self.ode.sorted_assignments(remove_unused=remove_unused)
        observed = self._observed(assignments)
        derivatives = [a for a in assignments if isinstance(a, atoms.StateDerivative)]
        equations = self._equations(derivatives)

        state_defaults = [(s.name, self._format_expr(s.value)) for s in self.ode.sorted_states()]
        param_defaults = [(p.name, self._format_expr(p.value)) for p in self.ode.parameters]
        param_defaults.extend((n, "0.0") for n in extra_params)

        parts = [
            self.template.header(),
            self.template.parameters_block(param_names, extra_params),
            self.template.states_block(state_names),
            self.template.observed_block(observed),
            self.template.equations_block(equations),
            self.template.defaults_block(state_defaults, param_defaults),
            self.template.ode_system(self.ode.name, state_names, param_names + extra_params),
        ]

        return "\n".join(parts)
