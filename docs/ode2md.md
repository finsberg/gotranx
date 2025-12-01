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

# Generating Markdown documentation

`gotranx` allows you to generate a Markdown file containing the mathematical representation of your ODE model. This is useful for creating readable documentation that can be rendered on platforms like GitHub or converted to PDF.

## Basic usage

To generate a Markdown file from an ODE file, you can use the `ode2md` command.

```bash
gotranx ode2md file.ode
```

This will create a file named `file.md` in the same directory. You can also specify the output filename using the `-o` or `--outname` option:

```bash
gotranx ode2md file.ode -o output.md
```

## Example

Consider the Lorentz system defined in `lorentz.ode`:

```
parameters(
sigma=12.0,
rho=21.0,
beta=2.4
)
states(x=1.0, y=2.0, z=3.05)

dx_dt = sigma * (y - x)
dy_dt = x * (rho - z) - y
dz_dt = x * y - beta * z
```

Let us write this to a file called `lorentz.ode`

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
with open("lorentz.ode", "w") as f:
    f.write(ode_string)
```

Running the following command:

```{code-cell} shell
!gotranx ode2md lorentz.ode
```

Will generate a `lorentz.md` file containing tables for parameters and states, as well as the equations formatted in LaTeX.

### Generated Output Preview

The generated Markdown will look something like this:

```{code-cell} shell
:tags: [scroll-output]

!cat lorentz.md
```

## Convert to pdf

If you would like to convert the `.ode` file directly to a PDF file, you can use the `--pdf` option:

```bash
gotranx ode2md lorentz.ode --pdf
```
This will create a `lorentz.pdf` file in the same directory. Note that this requires `pandoc` to be installed on your system.
