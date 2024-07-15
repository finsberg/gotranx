---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# About

In this document we will try to explain the process that is happening behind the scenes when an `.ode` file is parsed and code is generated.

To do this exercise we will consider the following ODE file called `lorentz.ode`

```{code-cell} python
ode_string = """
# This is the Lorentz system
# And it is part of this tutorial

parameters(
sigma=12.0,
rho=21.0,
beta=2.4
)

states(x=ScalarParam(1.0, unit="m", description="x variable"), y=2.0,z=3.05)

dx_dt = sigma * (y - x)  # The derivative of x
dy_dt = x * (rho - z) - y  # m/s
dz_dt = x * y - beta * z
"""

from gotranx.load import ode_from_string
ode = ode_from_string(ode_string, name="Lorentz")
print(ode)
```

To see what output is generated from this ODE you can check out [Your first ODE file](../examples/lorentz.py).

## Atoms

This ODE has 3 parameters (`sigma`, `rho` and `beta`), 3 states (`x`, `y` and `z`) and 3 assignments (`dx_dt`, `dy_dt` and `dz_dt`). Parameters, states and assignments are subclasses of `Atom` which is the most primitive building block in `gotranx`

```{mermaid}
classDiagram
    Atom <|-- Parameter
    Atom <|-- State
    Atom <|-- Assignment
    class Atom{
        name: str
        symbol: sympy.Symbol
        components: tuple[str, ...]
        description: str
        unit_str: str
        unit: pint.Unit

    }
    class Parameter{
        value: sympy.Number
    }
    class State{
        value: sympy.Number
    }
    class Assignment{
        value: Expression
        expr: sympy.Expression
        comment: Comment
    }
    Assignment <|-- StateDerivative
    Assignment <|-- Intermediate
    class Expression{
        tree: lark.Tree
        dependencies: set[str]
    }
    class StateDerivative{
        state: State
    }
```
An `Atom` contains a number of different fields

- `name` is the name of the variable represented as a string. For example of the name of the parameter `rho` is the string `"rho"`
- `symbol` is similar to `name` this this is a `sympy` object and this is used within the expressions in the assignments.
- `components` is just a list of components that a given atom belongs to. In this simple ODE we don't have any components (or practically speaking we have one component). However in lager ODE systems it might be useful to group parameters, states and assignments into different components. In these cases it is possible for an atom to be part of several components.
- `description` is just a string with some information. In our example, the state `x` has the description `"x variable"` and `dx_dt` has the description `"The derivative of x"`.
- `unit_str` is a string representation of a unit, for example `x` has the unit string `"m"` while `dy_dt` has the unit string `m/s`. If no using is provided this is set to `None`
- `unit` is a [pint unit](https://pint.readthedocs.io/en/stable/) and this can be used e.g to convert to and from different units.

We note that the 3 assignments are a special type of assignment called a `StateDerivative`. There is also another type of assignment called and `Intermediate`. The `StateDerivative` is special because it is associated with a given `State` and represents the temporal derivative of the state variable. For example `dx_dt` is associated with the state `x`.

Whenever you have a state variable, there has to be a corresponding state derivative. We can try to create an ODE with a missing derivative
```{code-cell} python
---
tags: [raises-exception]
---
from gotranx.load import ode_from_string

ode_from_string("states(x=1, y=2)\ndx_dt = x + y")
```

## Parsing an ODE file

When a file is parsed it is first opened, turned into a string and then sent to the [lark parser](https://lark-parser.readthedocs.io/en/stable/)

Lark then tries to parse the text using the [grammar](grammar.md) and turns each object into a tree which is in turn assembled into components using the {py:class}`transformer class<gotranx.transformer.TreeToODE>`.

```{mermaid}
flowchart LR
    FILE[.ode file] --> PARSER[Lark Parser]
    PARSER --> TRANS[Transformer]
    TRANS --> DESC[Description]
    TRANS --> TREE[Syntax Tree]
    ATOMS --> COMP[Components] --> ODE
    TRANS --> COMP
    TRANS --> ODE
    TREE --> ATOMS[Atom]

    DESC --> ODE
```

## Code generation

To generate code we pass the ODE to a code generator object. A code generator object inherits from the {py:class}`gotranx.codegen.CodeGenerator` class. For example we could create a PythonCodeGenerator by passing in the ode, i.e

```{code-cell} python
from gotranx.codegen.python import PythonCodeGenerator

codegen = PythonCodeGenerator(ode)
```

We can now generate code for the different methods. For example we can generate code for the `initial_state_values`

```{code-cell} python
print(codegen.initial_state_values())
```

Each code generator is associated with a `template` that follows the {py:class}`gotranx.templates.Template` protocol.


```{mermaid}
flowchart TD
    subgraph gen [Code generators]
    PYTHON_CODEGEN[python]
    C_CODEGEN[C]
    end
    ODE[ODE file] --> gen
    subgraph tem [Templates]
    PY_TEMP[Python template] --> PYTHON_CODEGEN
    C_TEMP[C template] --> C_CODEGEN
    end
    subgraph code [Code]
    PYTHON_CODEGEN -. Specific methods .-> PY_CODE[Python Code]
    C_CODEGEN -. Specific methods .-> C_CODE[C Code]
    end
```

You can checkout [Adding a new language](../examples/new-language/main.py) to see how to add support for a new language.
