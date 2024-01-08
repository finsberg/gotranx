from __future__ import annotations

# Copyright (C) 2013 Johan Hake
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

__all__ = ["MathMLBaseParser", "MathMLCPPParser"]
import sys
import structlog

logger = structlog.getLogger(__name__)

_all_keywords: list[str] = []


class MathMLBaseParser(object):
    def __init__(self):
        self._state_variable = None
        self._derivative = None
        self.variables_names = set()

        self._precedence = {
            "piecewise": 0,
            "power": 0,
            "rem": 0.5,
            "divide": 1,
            "times": 2,
            "minus": 4,
            "plus": 5,
            "lt": 6,
            "gt": 6,
            "leq": 6,
            "geq": 6,
            "eq": 10,
            "exp": 10,
            "ln": 10,
            "abs": 10,
            "floor": 10,
            "log": 10,
            "root": 10,
            "tan": 10,
            "cos": 10,
            "sin": 10,
            "tanh": 10,
            "cosh": 10,
            "sinh": 10,
            "arccos": 10,
            "arcsin": 10,
            "arctan": 10,
        }

        self._operators = {
            "power": "**",
            "rem": " % ",
            "divide": "/",
            "times": "*",
            "minus": " - ",
            "plus": " + ",
            "lt": " < ",
            "gt": " > ",
            "leq": " <= ",
            "geq": " >= ",
            "eq": " = ",
            "exp": "exp",
            "ln": "log",
            "abs": "abs",
            "floor": "floor",
            "log": "log",
            "root": "sqrt",
            "tan": "tan",
            "cos": "cos",
            "sin": "sin",
            "tanh": "tanh",
            "cosh": "cosh",
            "sinh": "sinh",
            "arccos": "acos",
            "arcsin": "asin",
            "arctan": "atan",
        }

    def use_parenthesis(self, child, parent, first_operand=True):
        """
        Return true if child operation need parenthesis
        """
        if parent is None:
            return False

        parent_prec = self._precedence[parent]
        if parent == "minus" and not first_operand:
            parent_prec -= 0.5

        if parent == "divide" and child == "times" and first_operand:
            return False

        if parent == "minus" and child == "plus" and first_operand:
            return False

        return parent_prec < self._precedence[child]

    def __getitem__(self, operator):
        return self._operators[operator]

    def _gettag(self, node):
        """
        Splits off the namespace part from name, and returns the rest, the tag
        """
        return "".join(node.tag.split("}")[1:])

    def parse(self, root):
        """
        Recursively parse a mathML subtree and return an list of tokens
        together with any state variable and derivative.
        """
        self._state_variable = None
        self._derivative = None
        self.used_variables = set()

        equation_list = self._parse_subtree(root)
        return (
            equation_list,
            self._state_variable,
            self._derivative,
            self.used_variables,
        )

    def _parse_subtree(self, root, parent=None, first_operand=True):
        op = self._gettag(root)

        # If the tag i "apply" pick the operator and continue parsing
        if op == "apply":
            children = list(root)
            op = self._gettag(children[0])
            root = children[1:]
        # If use special method to parse
        if hasattr(self, "_parse_" + op):
            return getattr(self, "_parse_" + op)(root, parent)
        elif op in list(self._operators.keys()):
            # Build the equation string
            eq = []

            # Check if we need parenthesis
            use_parent = self.use_parenthesis(op, parent, first_operand)

            # If unary operator
            if len(root) == 1:
                # Special case if operator is "minus"
                if op == "minus":
                    # If an unary minus is infront of a cn or ci we skip
                    # parenthesize
                    if self._gettag(root[0]) in ["ci", "cn"]:
                        use_parent = False

                    # If an unary minus is infront of a plus we always use parenthesize
                    if self._gettag(root[0]) == "apply" and self._gettag(
                        list(root[0])[0],
                    ) in ["plus", "minus"]:
                        use_parent = True

                    eq += ["-"]

                else:
                    # Always use paranthesis for unary operators
                    use_parent = True
                    eq += [self._operators[op]]

                eq += ["("] * use_parent + self._parse_subtree(root[0], op) + [")"] * use_parent
                return eq
            else:
                # Binary operator
                eq += ["("] * use_parent + self._parse_subtree(root[0], op)
                for operand in root[1:]:
                    eq = (
                        eq
                        + [self._operators[op]]
                        + self._parse_subtree(operand, op, first_operand=False)
                    )
                eq = eq + [")"] * use_parent

                return eq
        else:
            logger.error("No support for parsing MathML " + op + " operator.")

    def _parse_conditional(self, condition, operands, parent):
        return (
            [condition]
            + ["("]
            + self._parse_subtree(operands[0], parent)
            + [", "]
            + self._parse_subtree(operands[1], parent)
            + [")"]
        )

    def _parse_and(self, operands, parent):
        ret = ["And("]
        for operand in operands:
            ret += self._parse_subtree(operand, parent) + [", "]
        return ret + [")"]

    def _parse_or(self, operands, parent):
        ret = ["Or("]
        for operand in operands:
            ret += self._parse_subtree(operand, parent) + [", "]
        return ret + [")"]

    def _parse_lt(self, operands, parent):
        return self._parse_conditional("Lt", operands, "lt")

    def _parse_leq(self, operands, parent):
        return self._parse_conditional("Le", operands, "leq")

    def _parse_gt(self, operands, parent):
        return self._parse_conditional("Gt", operands, "gt")

    def _parse_geq(self, operands, parent):
        return self._parse_conditional("Ge", operands, "geq")

    def _parse_neq(self, operands, parent):
        return self._parse_conditional("Ne", operands, "neq")

    def _parse_eq(self, operands, parent):
        # Parsing conditional
        if parent == "piecewise":
            return self._parse_conditional("Eq", operands, "eq")

        # Parsing assignment
        return (
            self._parse_subtree(operands[0], "eq")
            + [self["eq"]]
            + self._parse_subtree(operands[1], "eq")
        )

    def _parse_pi(self, var, parent):
        return ["pi"]

    def _parse_ci(self, var, parent):
        varname = var.text.strip()
        if varname in _all_keywords:
            varname = varname + "_"
        self.used_variables.add(varname)
        return [varname]

    def _parse_cn(self, var, parent):
        value = var.text.strip()
        if "type" in list(var.keys()) and var.get("type") == "e-notation":
            # Get rid of potential float repr
            exponent = "e" + str(int(list(var)[0].tail.strip()))
        else:
            exponent = ""
        value += exponent

        return [value]

    def _parse_diff(self, operands, parent):
        # Store old used_variables so we can erase any collected state
        # variables
        used_variables_prior_parse = self.used_variables.copy()

        x = "".join(self._parse_subtree(operands[1], "diff"))
        y = "".join(self._parse_subtree(operands[0], "diff"))

        if x in _all_keywords:
            x = x + "_"

        if y == "time":
            y = "t"

        d = "d" + x + "_d" + y

        # Restore used_variables
        self.used_variables = used_variables_prior_parse

        # Store derivative
        self.used_variables.add(d)

        # This is an in/out variable remember it
        self._derivative = d
        self._state_variable = x
        return [d]

    def _parse_bvar(self, var, parent):
        if len(var) == 1:
            return self._parse_subtree(var[0], "bvar")
        else:
            logger.error("ERROR: No support for higher order derivatives.")

    def _parse_piecewise(self, cases, parent=None):
        if len(cases) == 2:
            piece_children = list(cases[0])
            cond = self._parse_subtree(piece_children[1], "piecewise")
            true = self._parse_subtree(piece_children[0])
            false = self._parse_subtree(list(cases[1])[0])
            return ["Conditional", "("] + cond + [", "] + true + [", "] + false + [")"]
        else:
            piece_children = list(cases[0])
            cond = self._parse_subtree(piece_children[1], "piecewise")
            true = self._parse_subtree(piece_children[0])
            return (
                ["Conditional", "("]
                + cond
                + [", "]
                + true
                + [", "]
                + self._parse_piecewise(cases[1:])
                + [")"]
            )


class MathMLCPPParser(MathMLBaseParser):
    def _parse_power(self, operands):
        return (
            ["pow", "("]
            + self._parse_subtree(operands[0])
            + [", "]
            + self._parse_subtree(operands[1])
            + [")"]
        )

    def _parse_piecewise(self, cases):
        if len(cases) == 2:
            piece_children = cases[0].getchildren()
            cond = self._parse_subtree(piece_children[1])
            true = self._parse_subtree(piece_children[0])
            false = self._parse_subtree(cases[1].getchildren()[0])
            return ["("] + cond + ["?"] + true + [":"] + false + [")"]
        else:
            sys.exit(
                "ERROR: No support for cases with other than two " "possibilities.",
            )
