from textwrap import dedent, indent


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
