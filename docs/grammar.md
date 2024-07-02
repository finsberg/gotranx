# Grammar

`gotranx` uses a domain specific language (DSL), that is implemented using [`lark`](https://github.com/lark-parser/lark), and you can find the complete grammar [here](https://github.com/finsberg/gotranx/blob/main/src/gotranx/ode.lark).

In this section, we will try to describe the grammar, and how you can write your own `.ode` files to define your ODEs using this grammar.

## Basic ODE file
An example `.ode` file called `lorentz.ode` could look as follows

```
# Define parameters
parameters(sigma=12.0, rho=21.0, beta=2.4)

# Define states
states(x=1.0, y=2.0, z=3.05)

# Define expressions
dx_dt = sigma * (y - x)
a = rho - z
dy_dt = x * a - y
dz_dt = x * y - beta * z
```

We will now go through the basic building block for creating an ODE

- `parameters` allows users to define parameters in the ODE. These are constant that the a user can provide as arguments. For example
    ```
    parameters(sigma=12.0, rho=21.0, beta=2.4)
    ```
    will create three parameters `sigma`, `rho` and `beta` with a default value of `12`, `21.0` and `2.4` respectively.

- `states` defines the state variables which are the variables you want to solve for. For example
  ```
  states(x=1.0, y=2.0, z=3.05)
  ```
  will define three state variables `x`, `y` and `z` who's default initial condition will be set to `1`, `2` and `3.0` respectively.

- `expressions` allows users to define mathematical expressions using parameters and other variables. You can also define new variables as an expression. In this section you also need to specify the the state derivatives of all the state variables. For the state derivatives you need the variables name to be `d{STATE_NAME}_dt`, e.g `dx_dt` for the state derivative of `x`. In this case we could for example have
  ```
  dx_dt = sigma * (y - x)
  a = rho - z
  dy_dt = x * a - y
  dz_dt = x * y - beta * z
  ```
  In this context, `a` will be referred to as an intermediate variable.

- Comments can be added on separate lines by starting the line `#`, for example
  ```
  # Define parameters
  ```

### Variables can only be defined once
An important thing to note is that once you have defined a variable you cannot redefine it, so for example the following expression is not valid
```
x = 1
y = 2 * x
x = 3
```

## Operators and symbols


### Mathematical operators

The following unary mathematical operators are supported

- `exp`
- `cos`
- `sin`
- `tan`
- `acos`
- `asin`
- `atan`
- `abs`
- `floor`
- `ln`
- `log`
- `sqrt`

The following binary mathematical operators are supported

- `Mod`

For example
```
x = abs(1.0 - exp(4))
```

### Numbers
A basic number can be written as e.g
```
x = 1             # Will be casted to an integer
y = 1.0           # Will be casted to a float
z = 1 / 4         # Will be cased to a rational number
w = (2 * 3) / 3   # Will be cased to an expression
```
It is also possible to use scientific notation using either uppercase of lowercase `e`, e.g the following numbers are equivalent
```
x = 100
x = 1e2
x = 1E2
```
Similarly you can also use negative exponents, e.g the following numbers are equivalent
```
x = 0.01
x = 1e-2
x = 1E-2
```

### Mathematical constants
There is currently only one reserved constant with is `pi`. This means that you can e.g write
```
x = cos(2 * pi)
```
without the need to define `pi` in advance

### Conditional operators and boolean operators

The following boolean operators are supported

- `Lt` which is equivalent to `<`
- `Gt` which is equivalent to `>`
- `Le` which is equivalent to `<=`
- `Ge` which is equivalent to `>=`
- `Eq` which is equivalent to `==`
- `Not`
- `And`
- `Or`

It is also possible to use the conditionals using the `Conditional` function (there is also a continuous version called `ContinuousConditional`), which has the following syntax
```
Conditional(boolean_expression, true_value, false_value)
```

For example the Heaviside step function

$$
H(x) = \begin{cases} 1, \;\; x \geq 0, \\ 0, \;\; x < 0, \end{cases}
$$

can be defined as follows
```
H = Conditional(Geq(x, 0), 1, 0)
```

This would be similar to implemented an `if-else` statement, e.g
```python
if x >= 0:
  H = 1
else:
  H = 0
```

We could also have a conditional with three value, e.g

$$
H(x) = \begin{cases} 1, \;\; x > 0, \\ 0.5 \;\; x = 0, \\ 0, \;\; x < 0, \end{cases}
$$

which would be equivalent to and `if-elif-else` statement, e.g
```python
if x > 0:
  H = 1
elif x == 0:
  H = 0.5
else:
  H = 0
```


but in this case we would need to nest conditionals together, for example

```
H = Conditional(Gt(x, 0), 1, Conditional(Eq(x, 0.0), 0.5, 0.0))
```
or another solution could be
```
H = Conditional(Gt(x, 0), 1, Conditional(Lt(x, 0.0), 0.0, 0.5))
```

### Implementing `min` and `max`

The operations `min` and `max` can be implemented in terms of `Conditional`, because
```python
f = min(x, y)
```
is equivalent to
```python
if x <= y:
  f = x
else:
  f = y
```
So `f = min(x, y)` could be implemented as
```
f = Conditional(Le(x, y), x, y)
```
Similarly `f = max(x, y)` can be implemented as
```
f = Conditional(Ge(x, y), x, y)
```


## Advanced usage

### Adding components, units and descriptions
If you have a large system of ODE's it might be good to group parts that together in the same group, which we will refer to as components.

```
states("membrane",
V=ScalarParam(-87, unit="mV", description="")
)

states("sodium_channel_h_gate",
h=ScalarParam(0.8, unit="1", description="")
)

states("sodium_channel_m_gate",
m=ScalarParam(0.01, unit="1", description="")
)

states("potassium_channel_n_gate",
n=ScalarParam(0.01, unit="1", description="")
)

parameters("membrane",
Cm=ScalarParam(12.0, unit="uF", description="")
)

parameters("leakage_current",
E_L=ScalarParam(-60.0, unit="mV", description=""),
g_L=ScalarParam(75.0, unit="uS", description="")
)

parameters("sodium_channel",
E_Na=ScalarParam(40.0, unit="mV", description=""),
g_Na_max=ScalarParam(400000.0, unit="uS", description="")
)

expressions("sodium_channel_h_gate")
alpha_h = 170.0*exp((-V - 1*90.0)/20.0) # S/F
beta_h = 1000.0/(exp((-V - 1*42.0)/10.0) + 1.0) # S/F
dh_dt = alpha_h*(1.0 - h) - beta_h*h

expressions("sodium_channel_m_gate")
alpha_m = (100.0*(-V - 1*48.0))/(exp((-V - 1*48.0)/15.0) - 1*1.0) # S/F
beta_m = (120.0*(V + 8.0))/(exp((V + 8.0)/5.0) - 1*1.0) # S/F
dm_dt = alpha_m*(1.0 - m) - beta_m*m

expressions("potassium_channel_n_gate")
alpha_n = (0.1*(-V - 1*50.0))/(exp((-V - 1*50.0)/10.0) - 1*1.0) # S/F
beta_n = 2.0*exp((-V - 1*90.0)/80.0) # S/F
dn_dt = alpha_n*(1.0 - n) - beta_n*n

expressions("potassium_channel")
g_K1 = 1200.0*exp((-V - 1*90.0)/50.0) + 15.0*exp((V + 90.0)/60.0) # uS
g_K2 = 1200.0*n**4.0 # uS
i_K = (V + 100.0)*(g_K1 + g_K2) # nA

expressions("sodium_channel")
g_Na = g_Na_max*(h*m**3.0) # uS
i_Na = (-E_Na + V)*(g_Na + 140.0) # nA

expressions("leakage_current")
i_Leak = g_L*(-E_L + V) # nA

expressions("membrane")
dV_dt = (-(i_Leak + (i_K + i_Na)))/Cm # mV
```
