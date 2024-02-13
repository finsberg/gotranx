from __future__ import annotations
from collections import defaultdict
import warnings
from pathlib import Path

import sympy as sp
import myokit
import myokit.formats.sympy
import myokit.lib.guess
import myokit.formats.cellml

from . import atoms
from .ode_component import MyokitComponent
from .ode import ODE

reserved_names = {name for name in dir(sp) if not name.startswith("_")}


def extract_unit(unit):
    if unit is None:
        return None
    return str(unit).strip("[").strip("]").replace("^", "**")


def extract_nested_variables(model):
    all_subs = defaultdict(dict)
    component_subs = defaultdict(dict)

    def f(component, all_subs, component_subs):
        component_subs_ = {}
        for var in component.variables():
            name = var.uname()
            if name in reserved_names:
                name = f"{name}_"

            component_subs[component.name()][sp.Symbol(var.name())] = sp.Symbol(name)
            all_subs[sp.Symbol(var.qname())] = sp.Symbol(name)
            all_subs, component_subs_ = f(var, all_subs, component_subs)
        component_subs.update(component_subs_)
        return all_subs, component_subs

    for component in model.components():
        all_subs, component_subs = f(component, all_subs, component_subs)

    return all_subs, component_subs


def mmt_to_gotran(filename: str | Path) -> ODE:
    model, protocol, _ = myokit.load(filename)

    return myokit_to_gotran(model, protocol=protocol)


def myokit_to_gotran(model: myokit.Model, protocol=None) -> ODE:
    # Embed protocol
    if protocol is not None:
        model = model.clone()
        if not myokit.lib.guess.add_embedded_protocol(model, protocol):
            warnings.warn("Unable to embed stimulus protocol.")

    model.create_unique_names()

    all_subs, component_subs = extract_nested_variables(model)

    initial_values = model.initial_values()
    components = []
    for component in model.components():
        states = []
        parameters = []
        intermediates = []
        derivatives = []
        for var in component.variables(deep=True):
            name = var.uname()
            if name in reserved_names:
                name = f"{name}_"

            if name == "time":
                # Skip time variable
                continue

            if var.is_state():
                state = atoms.State(
                    name=name,
                    value=initial_values[var.index()],
                    components=(component.name(),),
                    unit_str=extract_unit(var.unit()),
                    description=var.meta.get("desc", ""),
                )
                states.append(state)
                with sp.core.parameters.evaluate(False):
                    expr = myokit.formats.sympy.write(var.eq().rhs)
                    expr = expr.xreplace({v.name(): v.uname() for v in var.variables(deep=True)})
                    expr = expr.xreplace(component_subs.get(component.name(), {}))
                    expr = expr.xreplace(all_subs)

                state_der = atoms.StateDerivative(
                    name=f"d{name}_dt",
                    expr=expr,
                    value=None,
                    components=(component.name(),),
                    unit_str=extract_unit(var.unit()),
                    description=var.meta.get("desc", ""),
                    state=state,
                )
                derivatives.append(state_der)

            else:
                with sp.core.parameters.evaluate(False):
                    expr = myokit.formats.sympy.write(var.rhs())
                if expr.is_Number:
                    parameter = atoms.Parameter(
                        name=name,
                        value=var.value(),
                        components=(component.name(),),
                        unit_str=extract_unit(var.unit()),
                        description=var.meta.get("desc", ""),
                    )
                    parameters.append(parameter)

                else:
                    with sp.core.parameters.evaluate(False):
                        expr = expr.xreplace(
                            {v.name(): v.uname() for v in var.variables(deep=True)}
                        )
                        expr = expr.xreplace(component_subs.get(component.name(), {}))
                        expr = expr.xreplace(all_subs)

                    intermediate = atoms.Intermediate(
                        name=name,
                        expr=expr,
                        value=None,
                        components=(component.name(),),
                        unit_str=extract_unit(var.unit()),
                        description=var.meta.get("desc", ""),
                    )
                    intermediates.append(intermediate)

        comp = MyokitComponent(
            name=component.name(),
            states=frozenset(states),
            parameters=frozenset(parameters),
            intermediates=frozenset(intermediates),
            state_derivatives=frozenset(derivatives),
        )
        components.append(comp)

    return ODE(
        components=components,
        name=model.name(),
        comments=(atoms.Comment(model.meta.get("desc", "")),),
    )


def gotran_to_myokit(ode: ODE) -> myokit.Model:
    model = myokit.Model(ode.name)
    model.meta["author"] = "GotranX API"
    comp = model.add_component("engine")
    # breakpoint()
    time = comp.add_variable("time", binding="time")
    time.set_rhs(0)
    model._bindings["time"] = time
    # model._register_binding("time")

    # First we need to add all variables to the model
    global_var_map = {sp.Symbol("time"): sp.Symbol("engine.time")}
    for component in ode.components:
        comp = model.add_component(component.name)
        for state_derivative in component.state_derivatives:
            state = state_derivative.state
            var = comp.add_variable(state.name)
            global_var_map[sp.Symbol(state.name)] = sp.Symbol(var.qname())

        for parameter in component.parameters:
            var = comp.add_variable(parameter.name)
            var.set_rhs(parameter.value)
            global_var_map[sp.Symbol(parameter.name)] = sp.Symbol(var.qname())

        for intermediate in component.intermediates:
            var = comp.add_variable(intermediate.name)
            global_var_map[sp.Symbol(intermediate.name)] = sp.Symbol(var.qname())

    sympy_reader = myokit.formats.sympy.SymPyExpressionReader(model=model)
    # Then we can add expressions
    for component in ode.components:
        comp = model[component.name]

        for state_derivative in component.state_derivatives:
            state = state_derivative.state
            v = comp[state.name]

            expr = state_derivative.expr.xreplace(global_var_map)
            expr = sympy_reader.ex(expr)
            v.set_rhs(expr)

            v.promote(state.value)

        for intermediate in component.intermediates:
            v = comp[intermediate.name]
            expr = intermediate.expr.xreplace(global_var_map)
            expr = sympy_reader.ex(expr)
            v.set_rhs(expr)

    model.validate()
    return model


def cellml_to_gotran(filename: str | Path) -> ODE:
    myokit_model = myokit.formats.cellml.CellMLImporter().model(filename)
    return myokit_to_gotran(myokit_model)


def gotran_to_cellml(ode: ODE, filename: str | Path) -> None:
    model = gotran_to_myokit(ode)
    myokit.formats.cellml.CellMLExporter().model(filename, model, version="1.0")
