from __future__ import annotations
from textwrap import dedent


def parameter_table(parameters: list[dict[str, str]]) -> str:
    if not parameters:
        return ""

    header = """
| Name | Value | Unit | Description |
| :--- | :--- | :--- | :--- |"""

    rows = []
    for p in parameters:
        rows.append(f"| {p['name']} | {p['value']} | {p['unit']} | {p['description']} |")

    return header + "\n" + "\n".join(rows)


def state_table(states: list[dict[str, str]]) -> str:
    if not states:
        return ""

    header = """
| Name | Initial Value | Unit | Description |
| :--- | :--- | :--- | :--- |"""

    rows = []
    for s in states:
        rows.append(f"| {s['name']} | {s['value']} | {s['unit']} | {s['description']} |")

    return header + "\n" + "\n".join(rows)


def equation_block(equations: list[str]) -> str:
    if not equations:
        return ""

    eqs = "\n".join(equations)
    return dedent(f"""
$$
\\begin{{aligned}}
{eqs}
\\end{{aligned}}
$$
""")


def component_section(name: str, states: str, parameters: str, equations: str) -> str:
    return dedent(f"""
## Component: {name}

{parameters}

{states}

### Equations
{equations}
""")
