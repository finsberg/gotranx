from __future__ import annotations
from textwrap import dedent, indent
from structlog import get_logger

logger = get_logger()


def init_state_values(name, state_names, state_values, code):
    indented_code = indent(code, "    ")
    indent_values = indent(", ".join(f"{n}={v}" for n, v in zip(state_names, state_values)), "    ")
    return dedent(
        f"""
void init_state_values(double* {name}){{
    /*
{indent_values}
    */
{indented_code}
}}
""",
    )


def init_parameter_values(name, parameter_names, parameter_values, code):
    indented_code = indent(code, "    ")
    indent_values = indent(
        ", ".join(f"{n}={v}" for n, v in zip(parameter_names, parameter_values)), "    "
    )
    return dedent(
        f"""
void init_parameter_values(double* {name}){{
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
    **kwargs,
):
    indent_states = indent(states, "    ")
    indent_parameters = indent(parameters, "    ")
    indent_values = indent(values, "    ")
    return dedent(
        f"""
void {name}({args}){{

    // Assign states
{indent_states}

    // Assign parameters
{indent_parameters}

    // Assign expressions
{indent_values}
}}
""",
    )


def method_index(data: dict[str, int], method_name) -> str:
    logger.debug(f"Generating {method_name}_index with {len(data)} values")
    local_template = dedent(
        """
    {if_stm} (strcmp(name, "{name}") == 0) {{
        return {index};
    }}
    """
    )
    code = []
    code.append(f"// {method_name.capitalize()} index")
    code.append(f"int {method_name}_index(const char name[])")
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
    return method_index(data, "monitor")
