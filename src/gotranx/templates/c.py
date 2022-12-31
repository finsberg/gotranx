from textwrap import dedent

INIT_STATE_VALUES = dedent(
    """
void init_state_values(double* {name}){{
    /*
    {values}
    */
    {code}
}}
""",
)


INIT_PARAMETER_VALUES = dedent(
    """
void init_parameter_values(double* {name}){{
    /*
    {values}
    */
    {code}
}}
""",
)


RHS = dedent(
    """
void rhs({args}){{

    // Assign states
    {states}

    // Assign parameters
    {parameters}

    // Assign expressions
    {values}

}}
""",
)
