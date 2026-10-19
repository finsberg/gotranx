"""Templates for the C++ code generator.

The same functions as `templates.c`. Every function is `inline` (or `qualifier`), so the
header can be included from several translation units, and the index lookups use
`std::strcmp`.
"""

from __future__ import annotations
from textwrap import dedent, indent
from structlog import get_logger

logger = get_logger()


def init_state_values(
    name, state_names, state_values, code, float_type="double", qualifier="inline"
):
    indented_code = indent(code, "    ")
    indent_values = indent(", ".join(f"{n}={v}" for n, v in zip(state_names, state_values)), "    ")
    return dedent(
        f"""
{qualifier} void init_state_values({float_type}* {name}){{
    /*
{indent_values}
    */
{indented_code}
}}
""",
    )


def init_parameter_values(
    name, parameter_names, parameter_values, code, float_type="double", qualifier="inline"
):
    indented_code = indent(code, "    ")
    indent_values = indent(
        ", ".join(f"{n}={v}" for n, v in zip(parameter_names, parameter_values)), "    "
    )
    return dedent(
        f"""
{qualifier} void init_parameter_values({float_type}* {name}){{
    /*
{indent_values}
    */
{indented_code}
}}
""",
    )


def method(
    name: str,
    args: str,
    states: str,
    parameters: str,
    values: str,
    return_name: None = None,
    num_return_values: int = 0,
    missing_variables: str = "",
    qualifier: str = "inline",
    **kwargs,
):
    indent_states = indent(states, "    ")
    indent_parameters = indent(parameters, "    ")
    indent_values = indent(values, "    ")
    indent_missing_variables = indent(missing_variables, "    ")
    return dedent(
        f"""
{qualifier} void {name}({args}){{

    // Assign states
{indent_states}

    // Assign parameters
{indent_parameters}
{indent_missing_variables}
    // Assign expressions
{indent_values}
}}
""",
    )


def method_index(data: dict[str, int], method_name) -> str:
    logger.debug(f"Generating {method_name}_index with {len(data)} values")
    local_template = dedent(
        """
    {if_stm} (std::strcmp(name, "{name}") == 0) {{
        return {index};
    }}
    """
    )
    code = []
    code.append(f"// {method_name.capitalize()} index")
    code.append(f"inline int {method_name}_index(const char name[])")
    code.append("{")
    for i, (name, index) in enumerate(data.items()):
        if_stm = "if" if i == 0 else "else if"
        code.append(indent(local_template.format(if_stm=if_stm, name=name, index=index), "    "))
    code.append(indent("return -1;", "    "))
    code.append("}")

    return "\n".join(code)


def parameter_index(data: dict[str, int]) -> str:
    return method_index(data, "parameter")


def state_index(data: dict[str, int]) -> str:
    return method_index(data, "state")


def monitor_index(data: dict[str, int]) -> str:
    return method_index(data, "monitor")


def missing_index(data: dict[str, int]) -> str:
    return method_index(data, "missing")
