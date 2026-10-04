from __future__ import annotations

import structlog

from ..ode import ODE
from .. import templates
from .. import atoms
from .base import RHSArgument
from .julia import JuliaCodeGenerator

logger = structlog.get_logger()


class MissingPotentialState(ValueError):
    """The ODE has no state to use as the transmembrane potential."""


def _pascal_case(name: str) -> str:
    return "".join(
        part[:1].upper() + part[1:] for part in name.replace("-", "_").split("_") if part
    )


class CytoZooCodeGenerator(JuliaCodeGenerator):
    """Emit a CytoZoo cell-model adapter.

    Subclasses the Julia generator so every expression, every `TYPE(...)`-wrapped
    literal and every assignment block is shared rather than reimplemented. What
    differs is the RHS signature (Task 5), the parameter assignments (Task 5) and
    the adapter block below.
    """

    def __init__(
        self,
        ode: ODE,
        model_name: str | None = None,
        v_name: str = "v",
        remove_unused: bool = False,
        type_stable: bool = True,
    ) -> None:
        super().__init__(ode, remove_unused=remove_unused, type_stable=type_stable)
        self._model_name = model_name or _pascal_case(ode.name or "Model")
        self._v_name = v_name

    @property
    def template(self):
        return templates.julia

    @property
    def model_name(self) -> str:
        return self._model_name

    def _state_name_list(self) -> list[str]:
        # self.ode.states is name-sorted; sorted_states() is the dependency order
        # that fills `u`, which is what state_index()/_state_assignments() in
        # codegen/base.py actually use (base.py:171, base.py:268). Using the
        # name-sorted list here would silently mislabel every state slot.
        return [s.name for s in self.ode.sorted_states()]

    def _parameter_name_list(self) -> list[str]:
        return [p.name for p in self.ode.parameters]

    def _monitor_name_list(self) -> list[str]:
        # Mirror codegen.base.CodeGenerator.monitor_index (base.py:186) exactly:
        # one pass over sorted_assignments() in encounter order, not two
        # name-sorted blocks concatenated. Monitor slots must match monitor_index.
        return [
            a.name
            for a in self.ode.sorted_assignments(remove_unused=False)
            if isinstance(a, (atoms.Intermediate, atoms.StateDerivative))
        ]

    def _potential_index(self) -> int:
        names = self._state_name_list()
        try:
            return names.index(self._v_name) + 1
        except ValueError:
            raise MissingPotentialState(
                f"no state named {self._v_name!r} in ODE {self.ode.name!r}; "
                f"pass v_name=<state> (CLI: --v-name) naming the transmembrane potential. "
                f"States are: {', '.join(names)}"
            ) from None

    def _rhs_arguments(self, order=None, const_states: bool = True):
        """The Julia signature plus the two spatial arguments.

        `_cz_`-prefixed so an .ode declaring a parameter called `x` or `overrides`
        still generates valid code.
        """
        func = super()._rhs_arguments(
            order if order is not None else RHSArgument.tsp, const_states=const_states
        )
        return func._replace(arguments=list(func.arguments) + ["_cz_x", "_cz_overrides"])

    def _parameter_assignments(self, parameters) -> str:
        """Every parameter local resolves against the spatial overrides.

        `resolve_parameter` is CytoZoo public API. It is @inline and `name in names`
        folds against the override NamedTuple's type parameter, so a parameter with no
        override costs nothing at runtime; the ::Nothing method returns the fallback
        directly.
        """
        lines = []
        for i, param in enumerate(self.ode.parameters):
            if not self._condition(param.name):
                continue
            lines.append(
                f"{param.name} = TYPE(resolve_parameter("
                f"parameters[{i + 1}], _cz_overrides, Val(:{param.name}), _cz_x, t))"
            )
        return "\n".join(lines)

    def adapter(self) -> str:
        """The struct, name tuples, index lookups and interface methods."""
        name = self._model_name
        monitors = self._monitor_name_list()
        parts = [
            templates.cytozoo.model_struct(name, len(self.ode.parameters)),
            templates.cytozoo.name_tuples(
                name, self._state_name_list(), self._parameter_name_list(), monitors
            ),
            templates.cytozoo.index_lookups(
                name, self._state_name_list(), self._parameter_name_list()
            ),
            templates.cytozoo.interface_methods(name, self._potential_index(), len(monitors)),
        ]
        return self._format("\n".join(parts))
