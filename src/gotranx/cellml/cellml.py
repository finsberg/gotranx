from __future__ import annotations

# Copyright (C) 2011-2012 Johan Hake
#
# This file is part of Gotran.
#
# Gotran is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Gotran is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Gotran. If not, see <http://www.gnu.org/licenses/>.
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections import deque
from collections import OrderedDict
from functools import cmp_to_key
from pathlib import Path
from xml.etree import ElementTree
import structlog

logger = structlog.getLogger(__name__)

from .mathml import MathMLBaseParser


def cmp(a, b):
    return (a > b) - (a < b)


_all_keywords: list[str] = []


# Local imports

__all__ = ["CellMLParser"]

si_unit_map = {
    "ampere": "A",
    "becquerel": "Bq",
    "candela": "cd",
    "celsius": "gradC",
    "coulomb": "C",
    "dimensionless": "1",
    "farad": "F",
    "gram": "g",
    "gray": "Gy",
    "henry": "H",
    "hertz": "Hz",
    "joule": "J",
    "katal": "kat",
    "kelvin": "K",
    "kilogram": "kg",
    "liter": "l",
    "litre": "l",
    "molar": "M",
    "lumen": "lm",
    "lux": "lx",
    "meter": "m",
    "metre": "m",
    "mole": "mole",
    "newton": "N",
    "ohm": "Omega",
    "pascal": "Pa",
    "radian": "rad",
    "second": "s",
    "siemens": "S",
    "sievert": "Sv",
    "steradian": "sr",
    "tesla": "T",
    "volt": "V",
    "watt": "W",
    "weber": "Wb",
}

prefix_map = {
    "deca": "da",
    "hecto": "h",
    "kilo": "k",
    "mega": "M",
    "giga": "G",
    "tera": "T",
    "peta": "P",
    "exa": "E",
    "zetta": "Z",
    "yotta": "Y",
    "deci": "d",
    "centi": "c",
    "milli": "m",
    "micro": "u",
    "nano": "n",
    "pico": "p",
    "femto": "f",
    "atto": "a",
    "zepto": "z",
    "yocto": "y",
    None: "",
    "-3": "m",
}

ui = "UNINITIALIZED"


class Equation:
    """
    Class for holding information about an Equation
    """

    def __init__(self, name, expr, used_variables):
        self.name = name
        self.expr = expr
        self.used_variables = used_variables
        self.dependent_equations = []
        self.component = None

    def check_dependencies(self, equation):
        """
        Check Equation dependencies
        """
        assert isinstance(equation, Equation)

        if equation.name in self.used_variables:
            self.dependent_equations.append(equation)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Equation({self.name} = {''.join(self.expr)})"

    def __eq__(self, other):
        if not isinstance(other, Equation):
            return False
        return other.name == self.name and other.component == self.component

    def __hash__(self):
        return hash(self.name + self.component.name)


class Component:
    """
    Class for holding information about a CellML Component
    """

    def __init__(self, name, variables, equations, state_variables=None):
        self.name = name

        self.variable_info = {}

        self.state_variables = OrderedDict(
            (state, variables.pop(state, None)) for state in state_variables
        )

        for state, _info in list(self.state_variables.items()):
            self.variable_info[state] = _info
            self.variable_info["type"] = "state_variable"

        self.parameters = OrderedDict(
            (name, _info) for name, _info in list(variables.items()) if _info["init"] is not None
        )

        for param, _info in list(self.parameters.items()):
            self.variable_info[param] = _info
            self.variable_info["type"] = "parameter"

        self.derivatives = state_variables

        self.parent = None
        self.children = []

        self.dependencies = OrderedDict()
        self.used_in = OrderedDict()

        # Store equations
        self.sort_and_store_equations(equations)

        # Extract info
        dummy = dict(init=None, unit="1", private=True)
        for eq in self.equations:
            self.variable_info[eq.name] = variables.get(eq.name, dummy)
            self.variable_info[eq.name]["type"] = "equation"

        # Get used variables
        self.used_variables = set()
        for equation in self.equations:
            self.used_variables.update(equation.used_variables)

        # Remove dependencies on names defined by component
        self.used_variables.difference_update(equation.name for equation in self.equations)

        self.used_variables.difference_update(name for name in self.parameters)

        self.used_variables.difference_update(name for name in self.state_variables)

    def sort_and_store_equations(self, equations):
        # Check if reserved name for state derivativeis is used as equation
        # name
        derivative_names = [f"d{der}_dt" for der in self.derivatives]

        for eq in equations[:]:
            if eq.name in derivative_names and len(eq.expr) == 1 and eq.expr[0] == eq.name:
                equations.remove(eq)

        # Check internal dependencies
        for eq0 in equations:
            eq0.dependent_equations = []

            # Store component
            eq0.component = self
            for eq1 in equations:
                if eq0 == eq1:
                    continue
                eq0.check_dependencies(eq1)

        sorted_equations = []
        while equations:
            equation = equations.pop(0)
            if any(dep in equations for dep in equation.dependent_equations):
                equations.append(equation)
            else:
                sorted_equations.append(equation)

        # Store the sorted equations
        self.equations = sorted_equations

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name + f"<{len(self.state_variables)}>"

    def __repr__(self):
        return f"Component<{self.name}, {len(self.state_variables)}>"

    def __eq__(self, other):
        if not isinstance(other, Component):
            return False

        return other.name == self.name

    def check_dependencies(self, component):
        """
        Check components dependencies
        """
        assert isinstance(component, Component)

        if any(equation.name in self.used_variables for equation in component.equations):
            dep_equations = [
                equation for equation in component.equations if equation.name in self.used_variables
            ]

            # Register mutual dependencies
            self.dependencies[component] = dep_equations
            component.used_in[self] = dep_equations

            # Add logics for dependencies of all Equations.
            for other_equation in component.equations:
                for equation in self.equations:
                    if other_equation.name in equation.used_variables:
                        equation.dependent_equations.append(other_equation)

    def change_parameter_name(self, oldname, newname=None):
        """
        Change the name of a parameter
        Assume the name is only used locally within this component
        """
        assert oldname in self.parameters

        # If no newname is give we pick one based on the component name
        if newname is None:
            newname = oldname + "_" + self.name.split("_")[0]

        logger.warning(
            "Locally change parameter name '{0}' to '{1}' in " "component '{2}'.".format(
                oldname, newname, self.name
            ),
        )

        # Update parameters
        self.parameters = OrderedDict(
            (newname if name == oldname else name, value)
            for name, value in list(self.parameters.items())
        )

        # Update equations
        for eqn in self.equations:
            while oldname in eqn.expr:
                eqn.expr[eqn.expr.index(oldname)] = newname

        return newname

    def change_state_name(self, oldname, newname=None):
        """
        Change the name of a state
        Assume the name is only used locally within this component
        """
        assert oldname in self.state_variables

        # If no newname is give we pick one based on the component name
        if newname is None:
            newname = oldname + "_" + self.name.split("_")[0]

        logger.warning(
            "Locally change state name '{0}' to '{1}' in component " "'{2}'.".format(
                oldname, newname, self.name
            ),
        )

        # Update parameters
        self.state_variables = OrderedDict(
            (newname if name == oldname else name, value)
            for name, value in list(self.state_variables.items())
        )

        oldder = self.derivatives[oldname]
        newder = oldder.replace(oldname, newname)
        self.derivatives = OrderedDict(
            (newname if name == oldname else name, newder if value == oldder else value)
            for name, value in list(self.derivatives.items())
        )
        # Update equations
        for eqn in self.equations:
            while oldname in eqn.expr:
                eqn.expr[eqn.expr.index(oldname)] = newname
            while oldder in eqn.expr:
                eqn.expr[eqn.expr.index(oldder)] = newder
            if oldder == eqn.name:
                eqn.name = newder

        # Update derivative equation
        old_eq_name = f"d{oldname}_dt"
        new_eq_name = f"d{newname}_dt"
        self.variable_info[new_eq_name] = self.variable_info.pop(
            old_eq_name,
            dict(init=None, unit="1", private=True, type="equation"),
        )

        return newname

    def change_equation_name(self, oldname, newname=None):
        assert (
            oldname in self.variable_info and self.variable_info[oldname].get("type") == "equation"
        )

        # If no newname is give we pick one based on the component name
        if newname is None:
            newname = oldname + "_" + self.name.split("_")[0]

        logger.warning(
            "Locally change equation name '{0}' to '{1}' in " "component '{2}'.".format(
                oldname, newname, self.name
            ),
        )

        # Go through all equations and change the name used locally
        for ind, eqn in enumerate(self.equations[:]):
            while oldname in eqn.expr:
                eqn.expr[eqn.expr.index(oldname)] = newname

            # Update the actuall equation
            if eqn.name == oldname:
                eqn.name = newname
                self.equations[ind] = eqn

        # Update meta info
        self.variable_info[newname] = self.variable_info.pop(oldname)

        return newname

    def change_variable_name(self, oldname, newname=None):
        if oldname not in self.variable_info:
            logger.error(
                "Cannot change variable name. {0} is not a variable " "in component {1}".format(
                    oldname, self.name
                ),
            )
            return

        vartype = self.variable_info[oldname]["type"]

        assert vartype in ["state_variable", "parameter", "equation"]

        if vartype == "state_variable":
            return self.change_state_name(oldname, newname)
        if vartype == "parameter":
            return self.change_parameter_name(oldname, newname)

        return self.change_equation_name(oldname, newname)


class CellMLParser(object):
    """
    This module parses a CellML XML-file and converts it to PyCC code
    """

    @staticmethod
    def default_parameters():
        # Get the default parameters from the global parameters object
        return {
            "change_state_names": [],
            "grouping": "encapsulation",
        }

    def __init__(self, model_source, params=None, targets=None):
        """
        Arguments:
        ----------

        model_source: str
            Path or url to CellML file
        params : dict
            A dict with parameters for the
        targets : list, dict (optional)
            Components of the model to parse
        """

        targets = targets or []
        params = params or {}
        assert isinstance(model_source, (str, Path))
        assert isinstance(targets, (list, dict))
        self._params = self.default_parameters()
        self._params.update(params)

        # Open file or url
        try:
            with open(model_source, "r") as fp:
                self.cellml = ElementTree.parse(fp).getroot()
        except IOError:
            try:
                fp = urllib.request.urlopen(model_source)
                self.cellml = ElementTree.parse(fp).getroot()
            except Exception:
                logger.error("ERROR: Unable to open " + model_source)

        self.model_source = model_source
        self.name = self.cellml.attrib["name"]
        logger.info(f"Parsing CellML model: {self.name}")
        self.mathmlparser = MathMLBaseParser()
        self.cellml_namespace = self.cellml.tag.split("}")[0] + "}"
        self.parse_units()
        self.components = self.parse_components(targets)

        self.documentation = self.parse_documentation()

    def parse_documentation(self):
        """
        Parse the documentation of the article
        """
        namespace = "{http://cellml.org/tmp-documentation}"
        article = self.cellml.iter(namespace + "article")

        try:
            article = list(article)[0]
        except IndexError:
            return ""

        articleinfo = list(article.iter(namespace + "articleinfo"))[0]

        # Get title
        if articleinfo and articleinfo.iter(namespace + "title"):
            title = list(articleinfo.iter(namespace + "title"))[0].text
        else:
            title = ""

        # Get model structure comments
        for child in list(article):
            if child.attrib.get("id") == "sec_structure":
                content = []
                for par in child.iter(namespace + "para"):
                    # Get lines
                    splitted_line = deque(
                        ("".join(text.strip() for text in par.itertext())).split(" "),
                    )

                    # Cut them in lines which are not longer than 80 characters
                    ret_lines = []
                    while splitted_line:
                        line_stumps = []
                        line_length = 0
                        while splitted_line and (line_length + len(splitted_line[0]) < 80):
                            line_stumps.append(splitted_line.popleft())
                            line_length += len(line_stumps[-1]) + 1
                        ret_lines.append(" ".join(line_stumps))

                    content.extend(ret_lines)
                    content.append("\n")

                # Clean up content
                content = (
                    ("\n".join(cont.strip() for cont in content))
                    .replace("  ", " ")
                    .replace(" .", ".")
                    .replace(" ,", ",")
                )
                break
        else:
            content = ""

        if title or content:
            return f"{title}\n\n{content}"

        return ""

    def _gettag(self, node):
        """
        Splits off the namespace part from name, and returns the rest, the tag
        """
        return "".join(node.tag.split("}")[1:])

    def parse_units(self):
        """
        Parse any declared units in the model
        """
        collected_units = OrderedDict()
        unit_map = si_unit_map.copy()

        # Extend unit_map
        for cellml_pref, pref in list(prefix_map.items()):
            if cellml_pref:
                unit_map[cellml_pref + "molar"] = pref + "M"
                collected_units[cellml_pref + "molar"] = {pref + "M": (pref + "M", "1")}

        parsed_twice = []
        all_units = deque(self.get_iterator("units"))
        while all_units:
            units = all_units.popleft()
            unit_name = units.attrib["name"]

            if unit_name in unit_map:
                continue

            collected_parts = OrderedDict()
            for unit in list(units):
                if unit.attrib.get("multiplier"):
                    logger.warning(f"skipping multiplier in unit {units.attrib['name']}")
                if unit.attrib.get("multiplier"):
                    logger.warning(f"skipping multiplier in unit {units.attrib['name']}")
                cellml_unit = unit.attrib.get("units")

                prefix = prefix_map[unit.attrib.get("prefix")]
                exponent = unit.attrib.get("exponent", "1")
                if cellml_unit in si_unit_map:
                    abbrev = si_unit_map[cellml_unit]
                    name = prefix + abbrev
                    if exponent not in ["0", "1"]:
                        fullname = name + "**" + exponent
                    else:
                        fullname = name

                    collected_parts[name] = (fullname, exponent)
                elif cellml_unit in collected_units:
                    if prefix:
                        logger.warning(f"Skipping prefix of unit '{cellml_unit}'")
                    for name, (fullnam, part_exponent) in list(
                        collected_units[cellml_unit].items(),
                    ):
                        new_exponent = float(part_exponent) * float(exponent)

                        if new_exponent % 1.0 == 0.0:
                            new_exponent = str(int(new_exponent))
                        else:
                            new_exponent = str(new_exponent)

                        if new_exponent not in ["0", "1"]:
                            fullname = name + "**" + new_exponent
                        else:
                            fullname = name

                        collected_parts[name] = (fullname, exponent)

                elif units not in parsed_twice:
                    all_units.append(units)
                    parsed_twice.append(units)
                    break
                else:
                    logger.warning(
                        "Unknown unit '{0}' in {1}".format(cellml_unit, units["name"]),
                    )

            else:
                # Try change mole*l**-1 to mM...

                collected_units[unit_name] = collected_parts
                unit_map[unit_name] = "*".join(
                    fullname for fullname, exp in list(collected_parts.values())
                )

        # Store unit map
        self.units_map = unit_map

    def get_iterator(self, name, item=None):
        """
        Return an element tree iterator
        """

        item = item if item is not None else self.cellml
        return item.iter(self.cellml_namespace + name)

    def check_and_register_component_variables(
        self,
        comp,
        collected_states,
        collected_parameters,
        collected_equations,
    ):
        """
        Check if component variables are allready collected
        """

        # Check for duplication of states
        for name in list(comp.state_variables.keys()):
            der_name = f"d{name}_dt"
            if name in self._params["change_state_names"]:
                newname = name + "_" + comp.name.split("_")[0]
                comp.change_state_name(name, newname)
                name = newname

            # Check state name vs collected state names
            elif name in collected_states:
                state_comp = collected_states[name]
                logger.info(
                    "Same state name: '{0}' is used in component: '{1}' " "and '{2}'.".format(
                        name, comp.name, state_comp.name
                    ),
                )
                for change_comp in [comp, state_comp]:
                    if (
                        change_comp.state_variables[name]["private"]
                        and change_comp.variable_info[der_name]["private"]
                    ):
                        new_name = change_comp.change_state_name(name)
                        if change_comp == state_comp:
                            collected_states[new_name] = change_comp
                        break
                else:
                    logger.warning(
                        "Could not resolve duplicated state name {0} in "
                        "component {1} and {2}. None of them are private "
                        "to the components.".format(name, comp.name, state_comp.name),
                    )

            # Check state name vs collected parameter names
            elif name in collected_parameters:
                param_comp = collected_parameters[name]
                logger.info(
                    "State name: '{0}' from component '{1}' is used as "
                    "parameter in component '{2}'.".format(
                        name,
                        comp.name,
                        param_comp.name,
                    ),
                )

                # If parameter is private we change that
                if param_comp.parameters[name]["private"]:
                    new_name = param_comp.change_parameter_name(name)
                    collected_parameters.pop(name)
                    collected_parameters[new_name] = param_comp

                # Elseif the state variable is private we change that one
                elif (
                    comp.state_variables[name]["private"]
                    and comp.variable_info[der_name]["private"]
                ):
                    name = comp.change_state_name(name)

                else:
                    logger.warning(
                        "Could not resolve duplicated state and "
                        "parameter name {0} in component {1} and {2}. "
                        "None of them are private to the "
                        "components.".format(name, comp.name, param_comp.name),
                    )

            # Check state name vs collected equation names
            elif name in collected_equations:
                eq_comp = collected_equations[name]
                logger.info(
                    "State name '{0}' from component '{1}' is used as "
                    "parameter in component '{2}'.".format(
                        name,
                        comp.name,
                        eq_comp.name,
                    ),
                )

                # If state is private we change it
                if (
                    comp.state_variables[name]["private"]
                    and comp.variable_info[der_name]["private"]
                ):
                    name = comp.change_state_name(name)
                elif eq_comp.variable_info[name]["private"]:
                    new_name = eq_comp.change_equation_name(name)
                    collected_equations.pop(name)
                    collected_equations[new_name] = eq_comp

                else:
                    logger.warning(
                        "Could not resolve duplicated state and "
                        "equation name {0} in component {1} and {2}. "
                        "None of them are private to the "
                        "components.".format(name, comp.name, eq_comp.name),
                    )

            # Register the collected state
            collected_states[name] = comp

        # Check parameters
        for name in list(comp.parameters.keys()):
            # Check parameter name vs collected state names
            if name in collected_states:
                state_comp = collected_states[name]
                der_name = f"d{name}_dt"
                logger.info(
                    "Same parameter and state name: '{0}' is used in "
                    "component '{1}' and '{2}'.".format(
                        name,
                        comp.name,
                        state_comp.name,
                    ),
                )
                # If parameter is private we change that
                if comp.parameters[name]["private"]:
                    name = comp.change_parameter_name(name)
                elif (
                    state_comp.state_variables[name]["private"]
                    and state_comp.variable_info[der_name]["private"]
                ):
                    new_name = state_comp.change_state_name(name)
                    collected_states.pop(name)
                    collected_states[new_name] = state_comp

                else:
                    logger.warning(
                        "Could not resolve duplicated state name {0} in "
                        "component {1} and {2}. None of them are private "
                        "to the components.".format(name, comp.name, state_comp.name),
                    )

            # Check state name vs collected parameter names
            elif name in collected_parameters:
                param_comp = collected_parameters[name]
                logger.info(
                    "Parameter name '{0}' from component '{1}' is used as "
                    "parameter in component '{2}'.".format(
                        name,
                        comp.name,
                        param_comp.name,
                    ),
                )

                # If registered parameter is private we change that
                if param_comp.parameters[name]["private"]:
                    new_name = param_comp.change_parameter_name(name)
                    collected_parameters.pop(name)
                    collected_parameters[new_name] = param_comp

                # if the parameter is private we change that one
                elif comp.parameters[name]["private"]:
                    name = comp.change_parameter_name(name)

                else:
                    logger.warning(
                        "Could not resolve duplicated parameter names "
                        "{0} in component {1} and {2}. "
                        "None of them are private to the "
                        "components.".format(name, comp.name, param_comp.name),
                    )

            # Check state name vs collected equation names
            elif name in collected_equations:
                eq_comp = collected_equations[name]
                logger.info(
                    "Parameter name '{0}' from component '{1}' "
                    "is used as parameter in component '{2}'.".format(
                        name,
                        comp.name,
                        eq_comp.name,
                    ),
                )

                # If parameter is private we change it
                if comp.parameters[name]["private"]:
                    name = comp.change_parameter_name(name)
                elif eq_comp.variable_info[name]["private"]:
                    new_name = eq_comp.change_equation_name(name)
                    collected_equations.pop(name)
                    collected_equations[new_name] = eq_comp

                else:
                    logger.warning(
                        "Could not resolve duplicated parameter and "
                        "equation names {0} in component {1} and {2}. "
                        "None of them are private to the "
                        "components.".format(name, comp.name, eq_comp.name),
                    )
            collected_parameters[name] = comp

        # Check equation name
        for eq in comp.equations:
            name = eq.name

            # If the equation is private we do not care if it is duplicated
            # elsewhere. Risky...
            # if comp.variable_info[name]["private"]:
            #    collected_equations[name] = comp
            #    continue

            # If the equation will be broadcasted to another name we continue
            # print comp.name, name, self.new_variable_connections.get(comp.name, {}).keys()
            # if name in self.new_variable_connections.get(comp.name, {}).keys():
            #    continue

            # Check equation name vs collected state names
            if name in collected_states:
                state_comp = collected_states[name]
                der_name = f"d{name}_dt"
                logger.info(
                    "Same equation and state name '{0}' is used in "
                    "component '{1}' and '{2}'.".format(
                        name,
                        comp.name,
                        state_comp.name,
                    ),
                )
                # If equation is private we change that
                # if comp.variable_info[name]["private"]:
                #    name = comp.change_equation_name(name)
                if (
                    state_comp.state_variables[name]["private"]
                    and state_comp.variable_info[der_name]["private"]
                ):
                    new_name = state_comp.change_state_name(name)
                    collected_states.pop(name)
                    collected_states[new_name] = state_comp

                else:
                    logger.warning(
                        "Could not resolve duplicated state and "
                        "equation name {0} in component {1} and {2}. "
                        "None of them are private to the "
                        "components.".format(name, comp.name, state_comp.name),
                    )

            # Check equation name vs collected parameter names
            elif name in collected_parameters:
                param_comp = collected_parameters[name]
                logger.info(
                    "Equation name '{0}' from component '{1}' is used as "
                    "parameter in component '{2}'.".format(
                        name,
                        comp.name,
                        param_comp.name,
                    ),
                )

                # If registered parameter is private we change that
                if param_comp.parameters[name]["private"]:
                    new_name = param_comp.change_parameter_name(name)
                    collected_parameters.pop(name)
                    collected_parameters[new_name] = param_comp

                # If the parameter is private we change that one
                # elif comp.variable_info[name]["private"]:
                #    name = comp.change_equation_name(name)

                else:
                    logger.warning(
                        "Could not resolve duplicated parameter and "
                        "equation name {0} in component {1} and {2}. "
                        "None of them are private to the "
                        "components.".format(name, comp.name, param_comp.name),
                    )

            # Check equation name vs collected equation names
            elif name in collected_equations:
                eq_comp = collected_equations[name]
                logger.info(
                    "Equation name '{0}' from component '{1}' is used as "
                    "equation name in component '{2}'.".format(
                        name,
                        comp.name,
                        eq_comp.name,
                    ),
                )

                # If equation is private we change it
                # if comp.variable_info[name]["private"] and \
                #       eq_comp.variable_info[name]["private"]:
                #    warning("Both equations are private. We do not change "\
                #            "name on either one of them.")

                # elif eq_comp.variable_info[name]["private"]:
                #    new_name = eq_comp.change_equation_name(name)
                #    collected_equations.pop(name)
                #    collected_equations[new_name] = eq_comp
                #
                # else:
                #    warning("Could not resolve duplicated parameter and "\
                #            "equation names {0} in component {1} and {2}. "\
                #            "None of them are private to the "\
                #            "components.".format(name, comp.name, eq_comp.name))

            collected_equations[name] = comp

    def parse_imported_model(self):
        """
        Parse any imported models
        """

        components = OrderedDict()

        # Collect states and parameters to check for duplicates
        collected_states = dict()
        collected_parameters = dict()
        collected_equations = dict()

        # Import other models
        imports = self.get_iterator("import")
        if imports:
            logger.info("Parsing imported models:")
        for model in imports:
            import_comp_names = dict()

            for comp in self.get_iterator("component", model):
                import_comp_names[comp.attrib["component_ref"]] = comp.attrib["name"]

            model_parser = CellMLParser(
                model.attrib["{http://www.w3.org/1999/xlink}href"],
                targets=import_comp_names,
            )

            for comp in model_parser.components:
                components[comp.name] = comp

        # Extract states, parameters and equations
        for comp in list(components.values()):
            self.check_and_register_component_variables(
                comp,
                collected_states,
                collected_parameters,
                collected_equations,
            )

        return components, collected_states, collected_parameters, collected_equations

    def get_parents(self, grouping, element=None):
        """
        If group was used in the cellml use it to gather parent information
        about the components
        """

        # Collect component encapsulation
        def get_encapsulation(elements, all_parents, parent=None):
            children = {}
            for encap in elements:
                name = encap.attrib["component"]
                all_parents[name] = parent
                if list(encap):
                    nested_children = get_encapsulation(list(encap), all_parents, name)
                    children[name] = dict(children=nested_children, parent=parent)
                else:
                    children[name] = dict(children=None, parent=parent)

            return children

        encapsulations = dict()
        all_parents = dict()
        for group in self.get_iterator("group", element):
            children = list(group)

            if children and children[0].attrib.get("relationship") == grouping:
                encapsulations = get_encapsulation(children[1:], all_parents)

        # If no group information in cellml extract potential parent information
        # from component names
        if not all_parents:
            # Iterate over the components
            comp_names = [comp.attrib["name"] for comp in self.get_iterator("component", element)]

            for parent_name in comp_names:
                for name in comp_names:
                    if parent_name in name and parent_name != name:
                        all_parents[name] = parent_name

        return encapsulations, all_parents

    def parse_single_component(
        self,
        comp,
        collected_states,
        collected_parameters,
        collected_equations,
    ):
        """
        Parse a single component and create a Component object
        """
        comp_name = comp.attrib["name"]

        # Collect variables and equations
        variables = OrderedDict()
        equations = []
        state_variables = OrderedDict()
        # derivatives = []

        # Get variables that are used outside the component
        variables_used_in_connections = list(
            self.new_variable_connections.get(comp_name, dict()).keys(),
        ) + list(self.same_variable_connections.get(comp_name, dict()).keys())

        # Get variable and initial values
        for var in self.get_iterator("variable", comp):
            var_name = var.attrib["name"]
            if var_name in _all_keywords:
                var_name = var_name + "_"

            # Check connection and pair it with interface information to
            # check if a variable is public or not
            public = var_name in variables_used_in_connections and (
                var.attrib.get("public_interface") == "out"
                or var.attrib.get("private_interface") == "out"
            )

            # Store variables using initial and unit
            variables[var_name] = dict(
                init=var.attrib.get("initial_value"),
                unit=self.units_map[var.attrib.get("units")],
                private=not public,
                public_interface=var.attrib.get("public_interface"),
                private_interface=var.attrib.get("private_interface"),
            )

        # Get equations
        for math in comp.iter("{http://www.w3.org/1998/Math/MathML}math"):
            for eq in list(math):
                (
                    equation_list,
                    state_variable,
                    derivative,
                    used_variables,
                ) = self.mathmlparser.parse(eq)

                # Get equation name
                eq_name = equation_list[0]

                if eq_name in _all_keywords:
                    equation_list[0] = eq_name + "_"
                    eq_name = equation_list[0]

                # Discard collected equation name from used variables
                used_variables.discard(eq_name)

                assert re.findall(r"(\w+)", eq_name)[0] == eq_name
                assert equation_list[1] == self.mathmlparser["eq"]
                equations.append(Equation(eq_name, equation_list[2:], used_variables))

                # Do not register state variables twice
                if state_variable is not None and state_variable not in state_variables:
                    state_variables[state_variable] = derivative

        # Create Component
        comp = Component(comp_name, variables, equations, state_variables)

        # Collect and check variables
        self.check_and_register_component_variables(
            comp,
            collected_states,
            collected_parameters,
            collected_equations,
        )

        return comp

    def sort_components(self, components, sorted_once=False):
        if not sorted_once:
            logger.info("Sorting components with respect to dependencies")
        else:
            logger.info("Sorting components with respect to dependencies second time")

        # Check internal dependencies
        for comp0 in components:
            for comp1 in components:
                if comp0 == comp1:
                    continue
                comp0.check_dependencies(comp1)

        def simple_sort(components):
            components = deque(components)
            dependant_components = []
            sorted_components = []
            while components:
                component = components.popleft()

                # Chek for circular dependancy
                if dependant_components.count(component) > 4:
                    components.append(component)
                    break

                if any(dep in components for dep in component.dependencies):
                    components.append(component)
                    dependant_components.append(component)
                else:
                    sorted_components.append(component)

            return sorted_components, list(components)

        # Initial sorting
        sorted_components, circular_components = simple_sort(components)

        # If no circular dependencies
        if not circular_components:
            return sorted_components

        try:
            import networkx as nx
        except ImportError:
            logger.warning(
                "networkx could not be imported. Circular "
                "dependencies between components will not be sorted out.",
            )
            return sorted_components + circular_components

        # Collect zero and one dependencies
        zero_dep_equations = set()
        one_dep_equations = set()
        equation_map = {}

        # Gather zero dependent equations
        for comp in circular_components:
            for dep_comp, equations in list(comp.dependencies.items()):
                for equation in equations:
                    if not equation.dependent_equations and equation.name in comp.used_variables:
                        zero_dep_equations.add(equation)
                        equation_map[equation.name] = equation

        # Check for one dependency if that is the zero one
        one_dep_zero_dep = defaultdict(set)
        for comp in circular_components:
            for dep_comp, equations in list(comp.dependencies.items()):
                for equation in equations:
                    if (
                        len(equation.dependent_equations) == 1
                        and equation.name in comp.used_variables
                        and equation.dependent_equations[0] in zero_dep_equations
                    ):
                        equation_map[equation.name] = equation
                        one_dep_equations.add(equation)
                        one_dep_zero_dep[equation.name].add(
                            equation.dependent_equations[0].name,
                        )

        # Try to eliminate circular dependency
        # Extract dependent equation to a new component

        # Find ODE component. If not found create one
        for comp in components:
            if comp.name == self.name:
                ode_comp = comp
                break
        else:
            ode_comp = Component(self.name, {}, [], {})

        # Valid edges for removal
        valid_edges = [eq.name for eq in zero_dep_equations] + [eq.name for eq in one_dep_equations]

        G = nx.MultiDiGraph()
        G.add_nodes_from([comp.name for comp in components])

        # Build graph
        for comp in components:
            [
                G.add_edge(dep.name, comp.name, key=equation.name)
                for dep, equations in list(comp.dependencies.items())
                for equation in equations
            ]

        # collecte edges that breaks cycles
        cycle_breakers = []

        # Collect data over the best edge to remove
        edge_score = defaultdict(lambda: 0)
        edge_to_nodes = defaultdict(set)
        for cycle in nx.simple_cycles(G):
            local_breaker = []
            for n0, n1 in zip(cycle[:-1], cycle[1:]):
                if all(edge in valid_edges for edge in G[n0][n1]):
                    local_breaker.append([edge for edge in G[n0][n1]])
                    for edge in local_breaker[-1]:
                        edge_to_nodes[edge].add((n0, n1))
            cycle_breakers.append(local_breaker)
            for local_edges in local_breaker:
                for edge in local_edges:
                    edge_score[edge] += 1

        # Sort the edges we should remove by a score given by how many
        # cycles it breaks by being removed.
        edge_score = sorted((edge_score[edge], edge) for edge in edge_score)

        # Collect edges to be removed
        edge_removal = set()
        cycles_fixed = [False] * len(cycle_breakers)

        while not all(cycles_fixed) and edge_score:
            score, edge = edge_score.pop()

            # Remove this/these edge/edges this iteration
            local_removal = [edge]

            # If adding a one dep edge we need to also remove its dependent edge
            dep_removal = set()
            if edge in one_dep_zero_dep:
                local_removal.extend(one_dep_zero_dep[edge])
                dep_removal.update(one_dep_zero_dep[edge])

            for edge_remove in local_removal:
                # Iterate over the different cycles
                for i, local_breakers in enumerate(cycle_breakers):
                    # If the cycle is fixed
                    if cycles_fixed[i]:
                        continue

                    # Go through the collected valid breakers and pick the one
                    # with least edges first
                    for j, local_edges in enumerate(
                        sorted(
                            local_breakers,
                            key=cmp_to_key(lambda o0, o1: cmp(len(o0), len(o1))),
                        ),
                    ):
                        # If the removed edge is in the local edges
                        if edge_remove in local_edges:
                            local_edges.remove(edge_remove)
                            edge_removal.add(edge_remove)

                            # Check any dependent edges
                            for dep_edge in one_dep_zero_dep.get(edge_remove, []):
                                edge_removal.add(dep_edge)

                            if len(local_edges) == 0:
                                cycles_fixed[i] = True

                                break

        # If no edge sugested for removal we pick the zero dep equations
        if not edge_removal:
            if zero_dep_equations:
                edge_removal = [eq.name for eq in zero_dep_equations]
            elif one_dep_equations:
                edge_removal = [eq.name for eq in one_dep_equations]
            else:
                logger.warning("Could not sort out circular dependencies.")

        # Remove the edges from the graph
        for edge in edge_removal:
            for n0, n1 in edge_to_nodes[edge]:
                G.remove_edge(n0, n1, key=edge)

        # Move the marked edges (equations) from the relevant components
        removed_equations = {}
        for edge in edge_removal:
            eq = equation_map[edge]

            old_comp = eq.component
            ode_comp.equations.append(eq)
            old_comp.equations.remove(eq)
            ode_comp.variable_info[eq.name] = old_comp.variable_info.pop(eq.name)

            # Store changed
            removed_equations[eq] = old_comp

            # Transfer used_in to new component
            new_dependent_componets = OrderedDict()
            for dep_comp, equations in list(old_comp.used_in.items()):
                assert equations

                if eq not in equations or dep_comp == ode_comp:
                    continue

                # Remove dependency from old component and add it to the new
                if len(equations) == 1:
                    new_dependent_componets[dep_comp] = old_comp.used_in.pop(dep_comp)
                else:
                    new_dependent_componets[dep_comp] = [
                        old_comp.used_in[dep_comp].pop(equations.index(eq)),
                    ]

                # Change component dependencies
                if old_comp in dep_comp.dependencies and eq in dep_comp.dependencies[old_comp]:
                    if len(dep_comp.dependencies[old_comp]) == 1:
                        dep_comp.dependencies.pop(old_comp)
                    else:
                        dep_comp.dependencies[old_comp].remove(eq)

                if ode_comp not in dep_comp.dependencies:
                    dep_comp.dependencies[ode_comp] = [eq]
                else:
                    dep_comp.dependencies[ode_comp].append(eq)

            # Store new component to the extracted equation
            eq.component = ode_comp
            ode_comp.dependent_componets = new_dependent_componets

            if ode_comp not in sorted_components:
                sorted_components.insert(0, ode_comp)

        # Sort newly added equations
        ode_comp.sort_and_store_equations(ode_comp.equations)

        # Sort graph components and apply the sortings to the collected
        # CellML compoents
        sort_again = False
        try:
            sorted_components = list(nx.topological_sort(G))
            components.sort(key=lambda n0: sorted_components.index(n0.name))
            message = "To avoid circular dependency the following equations " "has been moved:"

        except nx.NetworkXUnfeasible as e:
            logger.warning("Topological sort failed: " + str(e))
            message = (
                "In a try to avoid circular dependency the following equations " "has been moved:"
            )
            sort_again = True

        # Insert the ODE component with extracted equations first
        components.insert(0, ode_comp)

        logger.warning(message)

        for eq, old_comp in list(removed_equations.items()):
            logger.warning(f"{eq.name} : from {old_comp.name} to {ode_comp.name} component")

        if sort_again:
            # Try rebuild the graph and make another topological sort
            G = nx.MultiDiGraph()
            G.add_nodes_from([comp.name for comp in components])

            # Build graph
            for comp in components:
                [
                    G.add_edge(dep.name, comp.name, key=equation.name)
                    for dep, equations in list(comp.dependencies.items())
                    for equation in equations
                ]

            try:
                sorted_components = list(nx.topological_sort(G))
                components.sort(key=lambda n0: sorted_components.index(n0.name))
            except nx.NetworkXUnfeasible as e:
                logger.warning("Topological sort failed a second time: " + str(e))

        return components

    def parse_components(self, targets):
        """
        Build a dictionary containing dictionarys describing each
        component of the cellml model
        """

        # First parse connections which is used to determine the interface of
        # each variable
        (
            self.new_variable_connections,
            self.same_variable_connections,
        ) = self.parse_connections()

        # Parse imported components
        (
            components,
            collected_states,
            collected_parameters,
            collected_equations,
        ) = self.parse_imported_model()

        # Get parent relationship between components
        encapsulations, all_parents = self.get_parents(self._params["grouping"])

        if targets:
            # If the parent information was not of type encapsulation
            # regather parent information
            if self._params["grouping"] != "encapsulation":
                encapsulations, dummy = self.get_parents("encapsulation")

            # Add any encapsulated components to the target list
            for target, new_target_name in list(targets.items()):
                if target in encapsulations:
                    for child in encapsulations[target]["children"]:
                        targets[child] = child.replace(target, new_target_name)

            target_parents = dict()

            # Update all_parents
            for comp_name, parent_name in list(all_parents.items()):
                if parent_name not in targets:
                    continue
                target_parents[targets[comp_name]] = targets[parent_name]

        # Iterate over the components
        for comp in self.get_iterator("component"):
            comp_name = comp.attrib["name"]

            # Only parse selected and non-empty components
            if (targets and comp_name not in targets) or len(list(comp)) == 0:
                continue

            # If targets provides a name mapping give the component a new name
            if targets and isinstance(targets, dict):
                new_name = targets[comp_name]
                comp.attrib["name"] = new_name
                comp_name = new_name

            # Store component
            components[comp_name] = self.parse_single_component(
                comp,
                collected_states,
                collected_parameters,
                collected_equations,
            )

        # Add parent information
        all_component_names = list(components.keys())
        for name, comp in list(components.items()):
            if targets:
                parent_name = target_parents.get(name)
            else:
                parent_name = all_parents.get(name)

            if parent_name:
                comp.parent = components[parent_name]

                # If parent name in child name, reduce child name length
                if parent_name in comp.name:
                    old_name = comp.name
                    new_name = old_name.replace(parent_name, "").strip("_")
                    if new_name not in all_component_names:
                        comp.name = new_name
                        all_component_names.remove(old_name)
                        all_component_names.append(new_name)

                components[parent_name].children.append(comp)

        # If we only extract a sub set of component we do not sort
        if targets:
            return list(components.values())

        # Before dependencies are checked we change names according to
        # variable mappings in the original CellML file
        if self.new_variable_connections:
            logger.info("\nRenaming variables through connections")

        for comp, variables in list(self.new_variable_connections.items()):
            # Iterate over old and new names
            for oldname, newnames in list(variables.items()):
                # Check that the direction of the connection is correct
                # If no variable info is registered for this comp or its
                # children it is the wrong direction

                # print
                # print comp, oldname
                # print components[comp].variable_info.keys()
                # for child in components[comp].children:
                #    print child.name, child.variable_info.keys()
                #    if oldname in child.variable_info.keys():
                #        print "HURRAY"
                #        print child.variable_info[oldname]

                if oldname not in components[comp].variable_info and all(
                    oldname not in child.variable_info for child in components[comp].children
                ):
                    continue

                # Try access var_info
                var_info = components[comp].variable_info.get(oldname)

                # If not we try components children
                if var_info is None:
                    for child in components[comp].children:
                        if oldname in child.variable_info:
                            var_info = child.variable_info[oldname]
                            break

                # If variable is not intended out
                if not (
                    var_info.get("public_interface") == "out"
                    or var_info.get("private_interface") == "out"
                ):
                    continue

                # Check if the oldname is used in any components
                old_name_used = self.same_variable_connections[comp].get(oldname)

                # If there are only one newname we change the name of the
                # original equation
                if len(newnames) == 1:
                    # If not old name used or old name only used in children and
                    # we then assume it is private to the component relationship
                    # parent-child
                    if not old_name_used or all(
                        components[other_comp] in components[comp].children
                        for other_comp in old_name_used
                    ):
                        # Get new name
                        newname = list(newnames.keys())[0]

                        # If oldname in component
                        if components[comp].variable_info.get(oldname) is not None:
                            components[comp].change_variable_name(oldname, newname)

                        if old_name_used:
                            for other_comp in old_name_used:
                                components[other_comp].change_variable_name(
                                    oldname,
                                    newname,
                                )

                        for child in components[comp].children:
                            if child.variable_info.get(oldname) is not None:
                                child.change_variable_name(oldname, newname)
                    else:
                        # FIXME: Add equation with name change to component
                        pass
                else:
                    # FIXME: Add equation with name change to component
                    pass

        # Add dependencies and sort the components accordingly
        return self.sort_components(list(components.values()))

    def parse_connections(self):
        new_variable_names = dict()
        same_variable_names = dict()

        for con in self.get_iterator("connection"):
            con_map = list(self.get_iterator("map_components", con))[0]
            comp1 = con_map.attrib["component_1"]
            comp2 = con_map.attrib["component_2"]

            for var_map in self.get_iterator("map_variables", con):
                var1 = var_map.attrib["variable_1"]
                var2 = var_map.attrib["variable_2"]

                if var1 != var2:
                    if comp1 not in new_variable_names:
                        new_variable_names[comp1] = {var1: defaultdict(list)}
                    elif var1 not in new_variable_names[comp1]:
                        new_variable_names[comp1][var1] = defaultdict(list)
                    new_variable_names[comp1][var1][var2].append(comp2)

                    # Register both directions
                    if comp2 not in new_variable_names:
                        new_variable_names[comp2] = {var2: defaultdict(list)}
                    elif var2 not in new_variable_names[comp2]:
                        new_variable_names[comp2][var2] = defaultdict(list)
                    new_variable_names[comp2][var2][var1].append(comp1)

                else:
                    if comp1 not in same_variable_names:
                        same_variable_names[comp1] = {var1: [comp2]}
                    elif var1 not in same_variable_names[comp1]:
                        same_variable_names[comp1][var1] = [comp2]
                    else:
                        same_variable_names[comp1][var1].append(comp2)

                    # Register both directions
                    if comp2 not in same_variable_names:
                        same_variable_names[comp2] = {var2: [comp1]}
                    elif var2 not in same_variable_names[comp2]:
                        same_variable_names[comp2][var2] = [comp1]
                    else:
                        same_variable_names[comp2][var2].append(comp1)

        return new_variable_names, same_variable_names

    def to_gotran(self):
        """
        Generate a gotran file
        """
        gotran_lines = []
        for docline in self.documentation.split("\n"):
            gotran_lines.append("# " + docline)

        if gotran_lines:
            gotran_lines.extend([""])

        # Add component info
        declaration_lines = []
        equation_lines = []

        def unders_score_replace(comp):
            new_name = comp.name.replace("_", " ")

            # If only 1 state it might be included in the name
            state_name = ""
            if len(comp.state_variables) == 1:
                state_name = list(comp.state_variables.keys())[0]
                if state_name in comp.name:
                    new_name = state_name.join(
                        part.replace("_", " ") for part in comp.name.split(state_name)
                    )

            # Captitalize first word
            single_words = new_name.split(" ")
            if len(single_words[0]) > 1 and "_" not in single_words[0]:
                single_words[0] = single_words[0][0].upper() + single_words[0][1:]

            # If first word is only a character we assume the first two
            # words to stick together
            elif (
                len(single_words[0]) == 1
                and len(single_words) > 1
                and single_words[0] != state_name
            ):
                single_words = [single_words[0] + "_" + single_words[1]] + single_words[2:]

            return " ".join(single_words)

        # Iterate over components and collect stuff
        for comp in self.components:
            names = deque([unders_score_replace(comp)])

            parent = comp.parent
            while parent is not None:
                names.appendleft(unders_score_replace(parent))
                parent = parent.parent

            comp_name = ", ".join(f'"{name}"' for name in names)

            # Collect initial state values
            if comp.state_variables:
                declaration_lines.append("")
                declaration_lines.append(f"states({comp_name},")
                for name, _info in list(comp.state_variables.items()):
                    if _info["unit"] != "1":
                        declaration_lines.append(
                            "       {0} = ScalarParam({1}" ', unit="{2}"),'.format(
                                name,
                                float(_info["init"]),
                                _info["unit"],
                            ),
                        )
                    else:
                        declaration_lines.append(f"       {name} = {_info['init']},")
                declaration_lines[-1] = declaration_lines[-1][:-1] + ")"

            # Collect initial parameters values
            if comp.parameters:
                declaration_lines.append("")
                declaration_lines.append(f"parameters({comp_name},")
                for name, _info in list(comp.parameters.items()):
                    if _info["unit"] != "1":
                        declaration_lines.append(
                            "           {0} = ScalarParam({1}" ', unit="{2}"),'.format(
                                name,
                                float(_info["init"]),
                                _info["unit"],
                            ),
                        )
                    else:
                        declaration_lines.append(
                            f"           {name} = {_info['init']},",
                        )
                declaration_lines[-1] = declaration_lines[-1][:-1] + ")"

            # Collect all intermediate equations
            if comp.equations:
                equation_lines.append("")
                equation_lines.append(f"expressions({comp_name})")
                """
                for eq in comp.equations:
                    expr = []
                    for eqi in eq.expr:
                        if eqi.isdigit():
                            # This converts integers to floats
                            expr.append(str(float(eqi)))
                        else:
                            expr.append(eqi)
                    eq.expr = expr
                """

                equation_lines.extend(
                    "{0} = {1}{2}".format(
                        eq.name,
                        "".join(eq.expr),
                        " # {0}".format(comp.variable_info[eq.name]["unit"])
                        if comp.variable_info[eq.name]["unit"] != "1"
                        else "",
                    )
                    for eq in comp.equations
                )

        gotran_lines.append(
            f"# gotran file generated by cellml2gotran from {self.model_source}",
        )
        gotran_lines.extend(declaration_lines)
        gotran_lines.extend(equation_lines)
        gotran_lines.append("")
        gotran_lines.append("")

        # Return joined lines
        return "\n".join(gotran_lines)
