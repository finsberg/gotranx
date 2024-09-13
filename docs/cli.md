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

# Using the command line interface

The primary usage of `gotranx` is through the command line interface. For this demonstration we will use a pre-made model that is hosted in the [CellML repository](https://models.physiomeproject.org/cellml). In particular we will be using the original [Noble model from 1962](https://models.physiomeproject.org/e/2a6/noble_1962.cellml/view) which is probably one of the simplest models for modeling cardiac cells.

First we download the model from the CellML repository by cloning the git repo
```shell
git clone https://models.physiomeproject.org/workspace/noble_1962
```
You could also visit the [model page](https://models.physiomeproject.org/e/2a6/noble_1962.cellml/view) and download the model manually. Once downloaded you will find a file called `noble_1962.cellml` inside it. Let us see what this file contains
```{code-cell} shell
:tags: [scroll-output]

!cat noble_1962.cellml
```
The `cellml` format is a format similar to XML.

## Converting from `.cellml` to `.ode`

We will now convert the `.cellml` file to a `.ode` file using the following command
```{code-cell} shell
!python3 -m gotranx cellml2ode noble_1962.cellml
```
This `cellml` converter is actually based on a different project called [`myokit`](https://github.com/myokit/myokit). `gotranx` allows for converting to and from `myokit` models and `myokit` allows for conversion to and from `cellml`.

Once the conversion is done, we see that a new file called `noble_1962.ode` has been created with the following content
```{code-cell} shell
:tags: [scroll-output]

!cat noble_1962.ode
```

This file contains the ODE representation that is used by `gotranx`. Note that you could also just write your own `.ode` file. Take a look at the [grammar](grammar.md) to learn more about the DSL.

## Generating source code
Now that we have a `.ode` file we can use this to generate source code in python (or C) that can be used to solve the ODE. To do this we can use the commands


`````{tab-set}
````{tab-item} Python
```shell
python3 -m gotranx ode2py noble_1962.ode
```
````

````{tab-item} C
Either to at `.c` file
```shell
python3 -m gotranx ode2c noble_1962.ode --to .c
```
or to a `.h` file
```shell
python3 -m gotranx ode2c noble_1962.ode --to .h
```
````
`````

Let us generate some code in python
```{code-cell} shell
!python3 -m gotranx ode2py noble_1962.ode
```

Now let us take a look at the generated code

```{code-cell} shell
:tags: [scroll-output]

!cat noble_1962.py
```


We wee the the code contains the following functions

- {py:func}`parameter_index<gotranx.templates.Template.parameter_index>`
- {py:func}`state_index<gotranx.templates.Template.state_index>`
- {py:func}`monitor_index<gotranx.templates.Template.monitor_index>`
- {py:func}`init_parameter_values<gotranx.templates.Template.init_parameter_values>`
- {py:func}`init_state_values<gotranx.templates.Template.init_state_values>`
- {py:func}`rhs<gotranx.templates.Template.method>`
- {py:func}`monitor<gotranx.templates.Template.method>`


You can click on each of them so see what is the purpose and use of them.

In the case of Python, the source code will be saved in a file called `noble_1962.py`, and we can solve use the code to solve the ODE as follows

```{code-cell} python
import noble_1962 as model
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

T = 5  # Unit is in seconds

# Get default initial state values but add a custom value for V
y = model.init_state_values(V=-86)
# Get default parameter values but add a custom value for Cm
p = model.init_parameter_values(Cm=11)
t = np.linspace(0, T, 1000)

res = solve_ivp(model.rhs, (0, T), y, t_eval=t, method="Radau", args=(p,))

V_index = model.state_index("V")
i_K_index = model.monitor_index("i_K")

monitor = model.monitor_values(res.t, res.y, p)
i_K = monitor[i_K_index, :]

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(res.t, res.y[V_index, :])
ax[0].set_title("Membrane potential")
ax[1].plot(res.t, i_K)
ax[1].set_title("Potassium current")
plt.show()
```

Here we have also used [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) for solving the initial value problem.


### Generating schemes for ODE
In the example above we only generated the right hand side (function `rhs`) which can be passed to `solve_ivp`, but it might be more appropriate to use a specific numerical scheme for solving the ODE. One example of a numerical scheme is the *forward euler*  scheme. Another popular scheme for solving cardiac cell models is the [Generalized Rush Larsen scheme](https://doi.org/10.1109/TBME.2009.2014739).

We can generate this scheme using the following command
```{code-cell} shell
!python3 -m gotranx ode2py noble_1962.ode --scheme generalized_rush_larsen -o noble_1962_grl.py
```
Here we also specify that the code should be saved to a new file called `noble_1962_grl.py` (just to not conflict with the existing file)

The file `noble_1962_grl.py` will now also contain the function
```python
def generalized_rush_larsen(states, t, dt, parameters): ...
```

and we can now solve the problem with the following program
```{code-cell} python
import noble_1962_grl as model
import numpy as np
import matplotlib.pyplot as plt

y = model.init_state_values()
p = model.init_parameter_values()
dt = 1e-4  # 0.1 ms
T = 5
t = np.arange(0, T, dt)

V_index = model.state_index("V")
V = [y[V_index]]

for ti in t[1:]:
    y = model.generalized_rush_larsen(y, ti, dt, p)
    V.append(y[V_index])

plt.plot(t, V)
plt.show()
```
