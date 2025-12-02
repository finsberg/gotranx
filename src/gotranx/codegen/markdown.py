from __future__ import annotations
import sympy
from sympy.printing.latex import LatexPrinter
from .. import atoms
from ..ode import ODE
from .. import templates


class GotranLatexPrinter(LatexPrinter):
    def _print_Symbol(self, expr):
        # Remove the (t) dependency for cleaner display if it's a function
        if isinstance(expr, sympy.Function):
            return super()._print_Symbol(sympy.Symbol(expr.name))
        return super()._print_Symbol(expr)


def print_derivative(name) -> str:
    return f"\\frac{{d {name}}}{{dt}}"


class MarkdownGenerator:
    def __init__(self, ode: ODE):
        self.ode = ode
        self._printer = GotranLatexPrinter(settings={"mul_symbol": " \\cdot "})

    def _format_atom(self, atom: atoms.Atom) -> dict[str, str]:
        unit = f"${sympy.latex(atom.unit)}$" if atom.unit else "-"
        desc = atom.description if atom.description else "-"
        val = self._printer.doprint(atom.value)
        # Wrap value in math mode if it looks like a number or expression
        if not val.startswith("$"):
            val = f"${val}$"

        return {
            "name": f"`{atom.name}`",
            "value": val,
            "unit": unit,
            "description": desc,
        }

    def _print_assignment(self, assignment: atoms.Assignment) -> str:
        if isinstance(assignment, atoms.StateDerivative):
            lhs = print_derivative(assignment.state.name)
        else:
            lhs = self._printer.doprint(assignment.symbol)

        rhs = self._printer.doprint(assignment.expr)
        return f"{lhs} &= {rhs} \\\\"

    def generate(self) -> str:
        parts = [f"# {self.ode.name}\n"]

        if self.ode.text:
            parts.append(self.ode.text + "\n")

        for component in self.ode.components:
            # Gather Parameters
            params_data = [self._format_atom(p) for p in component.parameters]
            params_table = templates.markdown.parameter_table(params_data)
            if params_table:
                params_table = "### Parameters" + params_table

            # Gather States
            states_data = [self._format_atom(s) for s in component.states]
            states_table = templates.markdown.state_table(states_data)
            if states_table:
                states_table = "### States" + states_table

            # Gather Equations (Assignments + State Derivatives)
            # We sort them to ensure consistent output, usually intermediates first then derivatives
            eq_strs = []

            # Intermediates
            for assignment in component.intermediates:
                eq_strs.append(self._print_assignment(assignment))

            # Derivatives
            for deriv in component.state_derivatives:
                eq_strs.append(self._print_assignment(deriv))

            eq_block = templates.markdown.equation_block(eq_strs)

            parts.append(
                templates.markdown.component_section(
                    name=component.name,
                    states=states_table,
                    parameters=params_table,
                    equations=eq_block,
                )
            )

        return "\n".join(parts)
