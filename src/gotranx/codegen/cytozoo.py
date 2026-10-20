from __future__ import annotations

import re

import structlog

from ..ode import ODE
from .. import templates
from .. import atoms
from ..exceptions import InvalidModelName, InvalidSupertype, MissingPotentialState
from ..templates.cytozoo import DEFAULT_SUPERTYPE
from .base import RHSArgument
from .julia import JuliaCodeGenerator

logger = structlog.get_logger()

# Both exceptions live in gotranx.exceptions alongside every other GotranxError,
# so `except exceptions.GotranxError` catches them. They stay importable from
# here, which is where they are raised and where callers already import them.
__all__ = [
    "DEFAULT_SUPERTYPE",
    "CytoZooCodeGenerator",
    "InvalidModelName",
    "InvalidSupertype",
    "MissingPotentialState",
]

_SUPERTYPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_!]*(\.[A-Za-z_][A-Za-z0-9_!]*)*")


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
        supertype: str = DEFAULT_SUPERTYPE,
    ) -> None:
        super().__init__(ode, remove_unused=remove_unused, type_stable=type_stable)
        name = model_name or _pascal_case(ode.name or "Model")
        # `_pascal_case` only splits on `-`/`_`, so an ODE named `1962_noble`,
        # `tentusscher panfilov` or `my.model` -- and any `--model-name` the
        # user passes -- can still start with a digit or carry a space or a
        # dot. Each emits `struct <invalid>{T <: AbstractVector}`, a Julia
        # parse error hundreds of lines away from its cause.
        if not name.isidentifier():
            raise InvalidModelName(name)
        self._model_name = name
        if not _SUPERTYPE_RE.fullmatch(supertype):
            raise InvalidSupertype(supertype)
        self._supertype = supertype
        self._v_name = v_name

    @property
    def template(self):
        return templates.julia

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supertype(self) -> str:
        return self._supertype

    def _state_name_list(self) -> list[str]:
        # self.ode.states is name-sorted; sorted_states() is the dependency order
        # that fills `u`, which is what state_index()/_state_assignments() in
        # codegen/base.py actually use (base.py:171, base.py:268). Using the
        # name-sorted list here would silently mislabel every state slot.
        return [s.name for s in self.ode.sorted_states()]

    def _parameter_name_list(self) -> list[str]:
        # The missing variables follow the parameters, in missing_index order, because
        # the adapter stores them in the parameter vector (templates.cytozoo.model_struct).
        missing = sorted(self._missing_variables, key=self._missing_variables.__getitem__)
        return [p.name for p in self.ode.parameters] + missing

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
                v_name=self._v_name, ode_name=self.ode.name, state_names=names
            ) from None

    def _rhs_arguments(self, order=None, const_states: bool = True):
        """The Julia signature plus the two spatial arguments.

        `_cz_`-prefixed so an .ode declaring a parameter called `x` or `overrides`
        still generates valid code.

        Typing the overrides follows the house pattern CytoZoo's own models use
        (docs/src/guides/implementing_a_model.md, "Adding Spatial Support"):
        giving them their own type parameter makes the method specialize on the
        override NamedTuple's type, so the `Nothing` case compiles the whole
        spatial branch away instead of leaving a runtime check behind. Julia
        does not specialize on an untyped argument that is only passed along.

        The guide spells that type parameter `F`, which cannot be used verbatim
        here: `F` is Faraday's constant in every cardiac model, and the emitted
        body assigns a local `F = TYPE(resolve_parameter(...))` for it, which
        Julia rejects with "local variable name \"F\" conflicts with a static
        parameter". The generated name is `_cz_F` for the same reason the
        arguments are `_cz_`-prefixed, and for the same reason the element type
        is `TYPE` rather than `T`.
        """
        func = super()._rhs_arguments(
            order if order is not None else RHSArgument.tsp, const_states=const_states
        )
        post = func.post_function_signature
        if post:
            # " where {TYPE, TIME, PARAM, OUT}" -> " where {TYPE, TIME, PARAM, OUT, _cz_F}"
            post = post.rstrip().removesuffix("}") + ", _cz_F}"
        else:
            post = " where {_cz_F}"
        return func._replace(
            arguments=list(func.arguments) + ["_cz_x", "_cz_overrides::_cz_F"],
            post_function_signature=post,
        )

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

    def _missing_variables_assignments(self) -> str:
        """Every missing-variable local, converted to the state's element type.

        The missing variables are trailing entries of the parameter vector (see
        `templates.cytozoo.model_struct`), so they are converted like the parameters:
        a model built in Float64 and evaluated on Float32 states would otherwise
        compute every term that reads one in Float64.
        """
        if not self._missing_variables:
            return ""
        lines = [
            f"{name} = TYPE(missing_variables[{index + 1}])"
            for name, index in self._missing_variables.items()
        ]
        return "\n".join(["", self._comment("Assign missing variables"), *lines, ""])

    def adapter(self) -> str:
        """The struct, name tuples, index lookups and interface methods."""
        name = self._model_name
        monitors = self._monitor_name_list()
        parts = [
            templates.cytozoo.model_struct(
                name,
                len(self.ode.parameters),
                len(self._missing_variables),
                self._supertype,
            ),
            templates.cytozoo.name_tuples(
                name, self._state_name_list(), self._parameter_name_list(), monitors
            ),
            templates.cytozoo.index_lookups(
                name, self._state_name_list(), self._parameter_name_list()
            ),
            templates.cytozoo.interface_methods(
                name, self._potential_index(), len(monitors), len(self._missing_variables)
            ),
        ]
        return self._format("\n".join(parts))
