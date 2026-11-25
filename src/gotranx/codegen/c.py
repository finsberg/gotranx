from __future__ import annotations
import enum
import typing
import structlog
from sympy.printing.c import C99CodePrinter
from sympy.codegen.ast import Assignment
import sympy

from ..ode import ODE
from .. import atoms, operations, templates
from ..exceptions import ReservedName
from .base import CodeGenerator, Func, RHSArgument, SchemeArgument

logger = structlog.get_logger()


class Format(str, enum.Enum):
    clang_format = "clang-format"
    none = "none"


def get_formatter(format: Format) -> typing.Callable[[str], str]:
    if format == Format.none:
        return lambda x: x
    elif format == Format.clang_format:
        # Do not use clang_format on Windows
        import platform

        if platform.system() == "Windows":
            logger.warning("Cannot apply clang-format on Windows")
            return lambda x: x
        try:
            import clang_format_docs
        except ImportError:
            logger.warning("Cannot apply clang-format, please install 'clang-format-docs'")
            return lambda x: x
        else:
            return clang_format_docs.clang_format_str
    else:
        raise ValueError(f"Unknown format: {format}")


def bool_to_int(expr: str) -> str:
    return expr.replace("false", "0").replace("true", "1")


class GotranCCodePrinter(C99CodePrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._settings["contract"] = False

    def _print_Float(self, flt):
        return self._print(str(float(flt)))

    def _print_Piecewise(self, expr):
        if isinstance(expr.args[0][0], Assignment):
            result = []
            lhs = super()._print(expr.args[0][0].lhs)
            result.append(f"{super()._print(lhs)} = ")
            all_lsh_equal = True
            for arg in expr.args:
                result.append(f"({super()._print(arg[1])}) ? ")
                result.append(f"{super()._print(arg[0].rhs)}")
                result.append(" : ")
                all_lsh_equal = all_lsh_equal and super()._print(arg[0].lhs) == lhs

            assert all_lsh_equal, "All assignments in Piecewise must have the same lhs"

            if super()._print(arg[1]) == "true":
                result = result[:-3]
                result.append(f"{super()._print(arg[0].rhs)}")
            else:
                raise ValueError("Last condition in Piecewise must be True")

            result.append(";")
            value = "".join(result)
        else:
            value = bool_to_int(super()._print_Piecewise(expr))

        return value


class CCodeGenerator(CodeGenerator):
    #: The floating point type of every state, parameter, value and local
    float_type = "double"
    #: How a pointer argument is declared free of aliasing
    restrict = "__restrict"
    # Derived from float_type and restrict in __init__, unless a subclass sets them itself
    variable_prefix = "const double "
    missing_variables_argument = "const double *__restrict missing_variables"

    #: Name of the array that holds the precomputed parameter-only intermediates
    constants_name = "constants"
    #: How the generated code names its language in error messages
    language_name = "C"

    def __init__(
        self,
        ode: ODE,
        format: Format = Format.clang_format,
        remove_unused: bool = False,
        precompute_constants: bool = False,
    ) -> None:
        super().__init__(ode, remove_unused=remove_unused)
        self.precompute_constants = precompute_constants
        self._constant_index = _parameter_only_intermediates(ode) if precompute_constants else {}
        if precompute_constants:
            self._check_constants_name()
        if not self._overridden("variable_prefix"):
            self.variable_prefix = f"const {self.float_type} "
        if not self._overridden("missing_variables_argument"):
            self.missing_variables_argument = (
                f"const {self.float_type} *{self.restrict} missing_variables"
            )
        self._printer = GotranCCodePrinter()
        setattr(self, "_formatter", get_formatter(format=format))

    @property
    def printer(self):
        return self._printer

    @property
    def constant_index(self) -> dict[str, int]:
        """The intermediates in the constants table and their index in it.

        Every intermediate that depends, recursively, only on parameters, in the order
        they are evaluated. Empty without `precompute_constants`.
        """
        return dict(self._constant_index)

    def _check_constants_name(self) -> None:
        ode = self.ode
        names = [
            x.name
            for x in [*ode.states, *ode.parameters, *ode.intermediates, *ode.state_derivatives]
        ] + list(ode.missing_variables)
        if self.constants_name in names:
            raise ReservedName(name=self.constants_name, backend=self.language_name)

    def _doprint(self, lhs, rhs, use_variable_prefix: bool = False) -> str:
        # With precompute_constants, a parameter-only intermediate is read from the table
        # that `init_constants` filled, instead of being computed again
        if use_variable_prefix and isinstance(lhs, sympy.Symbol):
            index = self._constant_index.get(lhs.name)
            if index is not None:
                table = sympy.IndexedBase(self.constants_name, shape=(len(self._constant_index),))
                read = table[index]
                return super()._doprint(lhs, read, use_variable_prefix=True)
        return super()._doprint(lhs, rhs, use_variable_prefix=use_variable_prefix)

    def init_constants(self) -> str:
        """The function that fills the constants table from the parameters.

        Empty without `precompute_constants`. Each parameter-only intermediate is computed
        exactly as `rhs` would compute it, in the same order, so reading it from the table
        gives the same value bit for bit.
        """
        if not self.precompute_constants:
            return ""
        parameters = sympy.IndexedBase("parameters", shape=(self.ode.num_parameters,))
        table = sympy.IndexedBase(self.constants_name, shape=(len(self._constant_index),))
        lines = []
        for x in self.ode.sorted_assignments(remove_unused=False):
            index = self._constant_index.get(x.name)
            if index is None:
                continue
            lines.append(CodeGenerator._doprint(self, x.symbol, x.expr, use_variable_prefix=True))
            lines.append(CodeGenerator._doprint(self, table[index], x.symbol))
        code = self.template.init_constants(
            args=", ".join(
                [self._pointer("parameters"), self._pointer(self.constants_name, const=False)]
            ),
            parameters=self._parameter_assignments(parameters),
            code="\n".join(lines),
            **self._init_value_kwargs(),
        )
        return self._format(code)

    def _parameter_arguments(self) -> list[str]:
        """The `parameters` argument, followed by `constants` with precompute_constants"""
        arguments = [self._pointer("parameters")]
        if self.precompute_constants:
            arguments.append(self._pointer(self.constants_name))
        return arguments

    @property
    def template(self):
        return templates.c

    def _overridden(self, name: str) -> bool:
        """Whether a subclass of CCodeGenerator sets the class attribute `name` itself."""
        for cls in type(self).__mro__:
            if cls is CCodeGenerator:
                return False
            if name in vars(cls):
                return True
        return False

    def _init_value_kwargs(self) -> dict[str, typing.Any]:
        return {"float_type": self.float_type}

    def _pointer(self, name: str, const: bool = True) -> str:
        prefix = "const " if const else ""
        return f"{prefix}{self.float_type} *{self.restrict} {name}"

    def imports(self) -> str:
        return self._format(
            "\n".join(
                [
                    "#include <math.h>",
                    "#include <string.h>\n",
                ]
            )
        )

    def _rhs_arguments(
        self, order: RHSArgument | str = RHSArgument.stp, const_states: bool = True
    ) -> Func:
        value = RHSArgument.get_value(order)
        argument_dict = {
            "s": [self._pointer("states", const=const_states)],
            "t": [f"const {self.float_type} t"],
            "p": self._parameter_arguments(),
        }
        argument_list = [a for v in value for a in argument_dict[v]] + [
            f"{self.float_type}* values"
        ]
        states = sympy.IndexedBase("states", shape=(self.ode.num_states,))
        parameters = sympy.IndexedBase("parameters", shape=(self.ode.num_parameters,))
        values = sympy.IndexedBase("values", shape=(self.ode.num_states,))

        return Func(
            arguments=argument_list,
            states=states,
            parameters=parameters,
            values=values,
            values_type="",
        )

    def _scheme_arguments(
        self,
        order: SchemeArgument | str = SchemeArgument.stdp,
        const_states: bool = True,
    ) -> Func:
        value = SchemeArgument.get_value(order)
        argument_dict = {
            "s": [self._pointer("states", const=const_states)],
            "t": [f"const {self.float_type} t"],
            "d": [f"const {self.float_type} dt"],
            "p": self._parameter_arguments(),
        }
        argument_list = [a for v in value for a in argument_dict[v]] + [
            f"{self.float_type}* values"
        ]
        states = sympy.IndexedBase("states", shape=(self.ode.num_states,))
        parameters = sympy.IndexedBase("parameters", shape=(self.ode.num_parameters,))
        values = sympy.IndexedBase("values", shape=(self.ode.num_states,))

        return Func(
            arguments=argument_list,
            states=states,
            parameters=parameters,
            values=values,
            values_type="",
        )


def _parameter_only_intermediates(ode: ODE) -> dict[str, int]:
    """Every intermediate that depends, recursively, only on parameters, with its index in
    the constants table, in the order the assignments are evaluated.

    A state, `t` or a missing variable is not a parameter, so anything that reads one,
    directly or through another intermediate, is not in the table.
    """
    lookup: dict[str, atoms.Atom] = {p.name: p for p in ode.parameters}
    lookup |= {s.name: s for s in ode.states}
    lookup |= {x.name: x for x in ode.sorted_assignments(remove_unused=False)}
    deps = {name: operations._dependencies(atom) for name, atom in lookup.items()}
    found = operations._parameter_only(lookup, deps)
    hoisted = [
        x.name
        for x in ode.sorted_assignments(remove_unused=False)
        if isinstance(x, atoms.Intermediate) and x.name in found
    ]
    return {name: index for index, name in enumerate(hoisted)}
