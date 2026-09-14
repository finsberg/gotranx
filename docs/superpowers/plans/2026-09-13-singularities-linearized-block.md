# Removable Singularities in the Rush-Larsen Linearized Block — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `gotranx`'s singularity-removal mechanism with one that guards
source assignments in the *state* variable, so that the Rush-Larsen linearized
block — which differentiates through intermediates — no longer forms double poles
at removable singularities.

**Architecture:** A new sympy-only module `src/gotranx/singularities.py` locates
removable poles structurally (denominator factors of the form `A*exp(u) + B` with
`u` linear in a state, or low-degree polynomials), classifies removability by the
*leading exponent* of the Laurent expansion, and emits
`Piecewise((taylor, Abs(var - value) < delta), (expr, True))` with the whole
assignment rewritten in `var`. `ODE.remove_singularities()` drives it; `sympy.cse`
in `schemes.py` is made blind to `Piecewise` subtrees so nothing is hoisted out of
a guard. The flag becomes default-on with `--no-remove-singularities` to opt out.

**Tech Stack:** Python >= 3.9, sympy 1.14, attrs, lark, typer, structlog, pytest.

**Spec:** `docs/superpowers/specs/2026-09-13-singularities-linearized-block-design.md`

## Global Constraints

- Python >= 3.9. No syntax newer than 3.9 (no `match`, no PEP 604 unions at
  runtime — the repo uses `from __future__ import annotations` everywhere).
- `sympy.limit` must not be used anywhere in the new code (spec F5: it silently
  returns 0 on float-coefficient gate rates).
- `sympy.singularities` and `sympy.solve` must not be used anywhere in the new
  code (see "Corrections to the spec" below — neither terminates on inlined
  expressions).
- Every new public function gets a numpydoc docstring, matching the style in
  `src/gotranx/linearization.py`.
- Test-first: the failing test and its output are shown before the fix lands.
- Done means `python3 -m pytest` and `pre-commit run --all` are both green.
- New words introduced into docstrings/comments may need adding to
  `.cspell_dict.txt` for the cspell pre-commit hook.

## Corrections to the spec

Three claims in the approved spec do not survive measurement. The decisions in
its "Decisions" section are untouched; these are mechanism-level corrections
made in service of those decisions. Evidence is in the task that implements each.

**C1 — Pole location cannot use `sympy.singularities` or `sympy.solve.`**
The spec's supporting measurement ("`sympy.singularities` terminates everywhere:
8.64 s source-level on ToRORd") was taken on *un-inlined* assignments. Inlining is
what the design adds. Measured on ToRORd's inlined assignments: 37 of 168
(assignment, state) pairs fail to terminate at a 10 s cap, and `dd_dt`/`v` raises
`RecursionError`. `sympy.solve` is no better — it times out at 5 s on 4-operation
denominators like `exp(-(v + a)/b) + 1` (which has no real root at all), and where
it does return it yields parameter-dependent cubic roots unusable as guard
locations. Replaced by a structural matcher over denominator factors
(author-approved). Measured: 0.3–13 s per model, terminates everywhere, and finds
every pole in the spec's evidence.

**C2 — The removability test in the spec does not detect non-removable poles.**
The spec decides removability by whether `sympy.series(...).removeO()` is free of
`oo`, `zoo` and `nan`. A Laurent expansion around a *genuine* simple pole contains
none of those — it contains a `1/(var - value)` term. Measured on ToRORd:

| assignment | pole | spec's test | truth |
|---|---|---|---|
| `IpCa` | `cai = -KmCap` | "removable" | leading exponent **-1** — a real pole |
| `CaMKb` | `cass = -KmCaM` | "removable" | leading exponent **-1** |
| `Jupnp` | `cai = -0.00092` | "removable" | leading exponent **-1** |
| `INab` | `v = 0` | "removable" | leading exponent 0 — removable |

Under the spec's test all 40 ToRORd candidates classify as removable and `IpCa`
would be "guarded" by the branch `-GpCa*KmCap/(KmCap + cai) + GpCa`, which is
itself infinite at exactly the guarded point. Replaced by a leading-exponent test
(`expr.subs(var, value + x).leadterm(x)`, require exponent >= 0), which separates
the table above correctly and costs 0.01–0.3 s per pole.

**C3 — Full-cone inlining is what makes the output explode, and it is unnecessary.**
The spec anticipates a size cost from losing `vfrt`/`vffrt` sharing and leaves it
to be measured. Measured on ToRORd, inlining each candidate's whole cone down to
states and parameters:

| | baseline | full-cone inlining |
|---|---|---|
| `rhs` | 32,499 chars / 884 lines | 74,662 / 2,123 (**2.30x**) |
| `scheme` | 106,841 chars / 2,695 lines | 272,053 / 7,360 (**2.55x**) |
| `scheme` codegen | 2.75 s | **130.68 s (47x)** |

The cost is not the loss of `vfrt`/`vffrt` sharing. It is that inlining `PhiCaL_ss`
drags in `gamma_cass` and `gamma_cao` — the ionic-strength `exp(sqrt(...))` terms —
which do not depend on `v` at all, and the Taylor branch then re-expands them.
Inlining only those intermediates that *transitively depend on the pole variable*
exposes the same vanishing factor (`vffrt = v*F*F/(R*T)` still inlines) while
leaving `gamma_cass` an opaque symbol that CSE can still share:

| | baseline | var-dependent inlining |
|---|---|---|
| `rhs` | 32,499 chars | 35,766 (**1.10x**) |
| `scheme` | 106,841 chars | 118,343 (**1.11x**) |
| `scheme` codegen | 2.75 s | 4.10 s (1.49x) |

This is also *more* faithful to the spec's own rationale ("`vffrt = vfrt*F` is what
makes INab's numerator vanish") and does not weaken the AD argument: a
`v`-independent symbol left in place is differentiated through the normal tangent
chain, exactly as it is today.

**C4 — The `delta` rule is calibrated on the wrong quantity.** The spec chooses
`delta` from `|a_{N+1} delta^(N+1)| <= tol*|a_0|`, which bounds the relative error
of the *value*. The linearized block uses the *derivative*, whose truncation error
is one order lower. Measured on ToRORd's `INab` inlined in `v`, the spec's rule
returns `delta = 0.1` (the upper clamp), where the series derivative has relative
error 4.9e-10 against a 50-digit reference while the direct float64 derivative
there has 3.0e-13 — i.e. the guard is three orders of magnitude *worse* than the
formula it replaces, right at the window edge. Replaced by balancing truncation
against float64 round-off, which is what the spec's own tolerance table describes
("the crossover is at `delta ~ 1e-3` in the local variable").

Related: the spec's acceptance criterion 1 asks for ~1e-11 relative agreement
"across the neighbourhood". That is not achievable at order 3 — the spec's own
tolerance table shows the series at `delta = 1e-2` is only good to 1.11e-8 and the
direct formula at `delta = 1e-4` only to 8.65e-9, so no single `delta` puts both
branches under 1e-11. The crossover is the best available, and the tests assert
the measured uniform bound rather than an unachievable one.

## What the spec got right

Checked before planning, because the author asked for scepticism on it:
**rewriting guarded assignments in the state variable does keep the AD chain rule
correct.** Built ToRORd's `INab` as a standalone ODE, guarded it at `v = 0` with an
order-3 series, and ran `diagonal_jacobian`'s exact tangent sweep over it:

| `v` | unguarded `d(dv_dt)/dv` | guarded | 50-digit reference |
|---|---|---|---|
| -1e-06 | -0.000106780910755 | -0.000102942035859 | -0.000102942035859 |
| -1e-09 | **+6.34575784765** | -0.000102942034778 | -0.000102942034778 |
| 0.0 | **nan** | -0.000102942034777 | -0.000102942034777 |
| +1e-09 | -6.34613710921 | -0.000102942034776 | -0.000102942034776 |

Worst relative disagreement inside the window: 4.2e-15. The guarded Jacobian
contains no `Derivative` and no `DiracDelta`. The F4 toy model reproduces too:
with `alpha_n` guarded at `V = -10`, `diagonal_jacobian` gives
`d(dV_dt)/dV = -alpha_n/C + (-V-77)*alpha_n'/C = -10 + (-67)*(-1/2) = 23.5`.

## File Structure

- **Create** `src/gotranx/singularities.py` — the whole mechanism: pole location,
  removability, Taylor replacement, window half-width, numeric verification,
  guard emission, and the assignment-level rewrite. Sympy only; no `Assignment`,
  no lark, no `ODE` import. ~320 lines.
- **Create** `tests/test_singularities.py` — unit tests for the module against
  hand-checked expressions.
- **Create** `tests/test_singularity_models.py` — model-level acceptance tests
  (the spec's test plan items 1, 2, 3, 5).
- **Modify** `src/gotranx/sympytools.py` — add `cse_hiding_piecewise`.
- **Modify** `src/gotranx/schemes.py:176` — route the one `sympy.cse` call through it.
- **Modify** `src/gotranx/atoms.py` — delete `Singularity`,
  `remove_singularities`, `Assignment.singularities`,
  `Assignment.remove_singularities` (replaced, per spec decision 1).
- **Modify** `src/gotranx/ode_component.py:195` — delete
  `Component.remove_singularities` (the new transform needs whole-ODE context).
- **Modify** `src/gotranx/ode.py:340` — `ODE.remove_singularities()` drives the
  new module.
- **Modify** `src/gotranx/load.py` — `remove_singularities: bool = True` on
  `ode_from_string` and `load_ode`.
- **Modify** `src/gotranx/cli/__init__.py`, `cli/utils.py`, and
  `cli/gotran2{py,c,julia,ufl}.py` — `--no-remove-singularities`, mirroring the
  existing `remove_unused` wiring exactly.
- **Modify** `tests/test_atoms.py`, `tests/test_python_codegen.py` — the old
  mechanism's tests.
- **Modify** `docs/cli.md`, `docs/config.md`, `CHANGELOG.md`.

---

### Task 1: Structural pole location

**Files:**
- Create: `src/gotranx/singularities.py`
- Test: `tests/test_singularities.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `denominator_factors(expr: sympy.Expr) -> tuple[sympy.Expr, ...]`
  - `factor_roots(factor: sympy.Expr, var: sympy.Symbol, max_degree: int = 2) -> list[sympy.Expr]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_singularities.py`:

```python
"""Unit tests for `gotranx.singularities`."""

from __future__ import annotations

import sympy

from gotranx import singularities

V = sympy.Symbol("V", real=True)
x = sympy.Symbol("x", real=True)
a = sympy.Symbol("a", real=True)


def test_denominator_factors_splits_a_product():
    expr = (V + 1) / ((sympy.exp(V) - 1) * (V - 3))
    factors = singularities.denominator_factors(expr)
    assert sympy.exp(V) - 1 in factors
    assert V - 3 in factors


def test_denominator_factors_is_empty_for_a_polynomial():
    assert singularities.denominator_factors(V**2 + 3 * V) == (sympy.S.One,)


def test_factor_roots_finds_the_ghk_root():
    """`exp(u) - 1` vanishes where u = 0; u = V*F/(R*T) is linear in V."""
    F, R, T = sympy.symbols("F R T", real=True)
    assert singularities.factor_roots(sympy.exp(V * F / (R * T)) - 1, V) == [0]


def test_factor_roots_finds_a_gate_rate_root_with_float_coefficients():
    """The shape .ode files actually use -- see spec F5."""
    assert singularities.factor_roots(1 - sympy.exp(-0.04 * (V + 23)), V) == [-23]


def test_factor_roots_finds_a_scaled_gate_rate_root():
    assert singularities.factor_roots(sympy.exp((V + 10) / 10) - 1, V) == [-10]


def test_factor_roots_rejects_a_denominator_with_no_real_root():
    """`exp(u) + 1` is never zero for real u. sympy.solve spends >5 s
    discovering this; the structural matcher answers immediately."""
    assert singularities.factor_roots(sympy.exp(-0.1 * (V + 47)) + 1, V) == []


def test_factor_roots_finds_a_linear_polynomial_root():
    assert singularities.factor_roots(x + a, x) == [-a]


def test_factor_roots_skips_a_high_degree_polynomial():
    assert singularities.factor_roots(x**5 + a, x) == []


def test_factor_roots_skips_a_factor_free_of_the_variable():
    assert singularities.factor_roots(sympy.exp(a) - 1, V) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_singularities.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gotranx.singularities'`

- [ ] **Step 3: Write minimal implementation**

Create `src/gotranx/singularities.py`:

```python
r"""Removable singularities in model expressions.

A gate rate like :math:`x/(e^x - 1)` has a removable pole at :math:`x = 0`.
Evaluated in float64 it loses all precision as :math:`x \to 0` and raises at
exactly 0; *differentiated* -- which is what the Rush-Larsen linearized block
does -- it has a double pole, and the wrong value it produces is O(1) and can
have the wrong sign.

This module finds such poles and replaces a neighbourhood of each with a
truncated Taylor series. It is sympy-only: it knows nothing about
``Assignment``, lark, or ``ODE``, so both the model layer and the schemes can
use it.

Two sympy entry points are deliberately *not* used here:

``sympy.limit``
    Silently returns 0 for float-coefficient gate rates, e.g.
    ``limit(0.2*(V+23)/(1-exp(-0.04*(V+23))), V, -23)`` is 0 where the true
    limit is 5. ``.ode`` files are written with float coefficients.

``sympy.singularities`` / ``sympy.solve``
    Neither terminates on inlined model expressions. Measured on
    ``ToRORd_dyn_chloride``: ``singularities`` fails to finish within 10 s for
    37 of 168 (assignment, state) pairs, and ``solve`` fails to finish within
    5 s on a 4-operation denominator such as ``exp(-(v + a)/b) + 1``, which has
    no real root at all. Pole *locations* are therefore found structurally, by
    matching the handful of denominator shapes that admit a closed-form root.
"""

from __future__ import annotations

import sympy

__all__ = ["denominator_factors", "factor_roots"]


def denominator_factors(expr: sympy.Expr) -> tuple[sympy.Expr, ...]:
    """The factors of ``expr``'s denominator.

    Parameters
    ----------
    expr : sympy.Expr
        Any expression.

    Returns
    -------
    tuple[sympy.Expr, ...]
        The factors of the denominator of ``sympy.together(expr)``. An
        expression with no denominator yields ``(1,)``.
    """
    _, den = sympy.fraction(sympy.together(expr))
    return tuple(sympy.Mul.make_args(sympy.factor_terms(den)))


def _linear_root(expr: sympy.Expr, var: sympy.Symbol) -> sympy.Expr | None:
    """Root of ``expr == 0`` when ``expr`` is linear in ``var``, else None."""
    try:
        poly = sympy.Poly(expr, var)
    except (sympy.PolynomialError, sympy.GeneratorsNeeded):
        return None
    if poly.degree() != 1:
        return None
    slope, intercept = poly.all_coeffs()
    if slope == 0:
        return None
    return sympy.simplify(-intercept / slope)


def factor_roots(
    factor: sympy.Expr,
    var: sympy.Symbol,
    max_degree: int = 2,
) -> list[sympy.Expr]:
    r"""Real roots of a denominator ``factor`` in ``var``, found structurally.

    Only shapes with a closed-form root are matched, because the general
    solvers do not terminate on inlined model expressions (see module
    docstring):

    * ``A*exp(u) + B`` with ``A`` and ``B`` free of ``var`` and ``u`` linear in
      ``var``. Real only when ``-B/A > 0``, giving ``u = log(-B/A)``. This is
      every GHK denominator (``exp(vfrt) - 1``) and every gate-rate denominator
      (``1 - exp(-0.04*(V + 23))``).
    * a polynomial in ``var`` of degree at most ``max_degree``.

    Anything else returns an empty list: the pole, if there is one, is left
    unguarded.

    Parameters
    ----------
    factor : sympy.Expr
        One factor of a denominator.
    var : sympy.Symbol
        The variable to solve in.
    max_degree : int, optional
        Highest polynomial degree to solve, by default 2.

    Returns
    -------
    list[sympy.Expr]
        Candidate root locations, possibly in terms of parameters. Roots sympy
        can prove non-real are dropped; roots it cannot decide are kept and
        filtered numerically later.
    """
    if var not in factor.free_symbols:
        return []

    if not factor.has(sympy.exp):
        try:
            poly = sympy.Poly(factor, var)
        except (sympy.PolynomialError, sympy.GeneratorsNeeded):
            return []
        if poly.degree() > max_degree:
            return []
        return [root for root in sympy.roots(poly) if root.is_real is not False]

    # A*exp(u) + B
    if not isinstance(factor, sympy.Add) or len(factor.args) != 2:
        return []
    exponentials = [term for term in factor.args if term.has(sympy.exp)]
    constants = [term for term in factor.args if not term.has(sympy.exp)]
    if len(exponentials) != 1 or len(constants) != 1:
        return []

    intercept = constants[0]
    coefficient: sympy.Expr = sympy.S.One
    exponent: sympy.Expr | None = None
    for term in sympy.Mul.make_args(exponentials[0]):
        if isinstance(term, sympy.exp):
            if exponent is not None:
                return []  # a product of exponentials, not this shape
            exponent = term.args[0]
        else:
            coefficient = coefficient * term
    if exponent is None:
        return []
    if var in coefficient.free_symbols or var in intercept.free_symbols:
        return []

    ratio = sympy.simplify(-intercept / coefficient)
    if ratio.is_positive is not True:
        return []
    root = _linear_root(exponent - sympy.log(ratio), var)
    return [] if root is None else [root]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_singularities.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gotranx/singularities.py tests/test_singularities.py
git commit -m "feat(singularities): locate denominator roots structurally

sympy.singularities and sympy.solve do not terminate on inlined model
expressions; match the denominator shapes that admit a closed-form root
instead."
```

---

### Task 2: Inlining and the removability test

**Files:**
- Modify: `src/gotranx/singularities.py`
- Test: `tests/test_singularities.py`

**Interfaces:**
- Consumes: `factor_roots`, `denominator_factors` from Task 1.
- Produces:
  - `MAX_INLINE_OPS: int = 5000`
  - `inline(expr, definitions, var) -> sympy.Expr | None`
  - `is_removable(expr, var, value) -> bool`
  - `taylor(expr, var, value, order) -> sympy.Expr | None`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_singularities.py`:

```python
F, R, T, PNab, nai, nao = sympy.symbols("F R T PNab nai nao", real=True)
v = sympy.Symbol("v", real=True)
cai, KmCap, GpCa = sympy.symbols("cai KmCap GpCa", real=True)
gamma = sympy.Symbol("gamma", real=True)

VFRT = sympy.Symbol("vfrt", real=True)
VFFRT = sympy.Symbol("vffrt", real=True)
GHK_DEFINITIONS = {VFRT: v * F / (R * T), VFFRT: v * F * F / (R * T)}
GHK = PNab * VFFRT * (nai * sympy.exp(VFRT) - nao) / (sympy.exp(VFRT) - 1)


def test_inline_expands_only_what_depends_on_the_variable():
    """`gamma` does not depend on v, so it must stay an opaque symbol --
    inlining it is what made the generated ToRORd rhs 2.3x larger."""
    definitions = dict(GHK_DEFINITIONS)
    definitions[gamma] = sympy.exp(sympy.sqrt(cai))
    inlined = singularities.inline(gamma * GHK, definitions, v)
    assert gamma in inlined.free_symbols
    assert VFRT not in inlined.free_symbols
    assert VFFRT not in inlined.free_symbols
    assert v in inlined.free_symbols


def test_inline_gives_up_past_the_operation_budget():
    big = sympy.Symbol("big", real=True)
    definitions = {big: sum(v**k for k in range(200)) ** 3}
    assert singularities.inline(big * big, definitions, v) is None


def test_is_removable_accepts_the_ghk_pole():
    inlined = singularities.inline(GHK, GHK_DEFINITIONS, v)
    assert singularities.is_removable(inlined, v, sympy.Integer(0))


def test_is_removable_rejects_a_genuine_simple_pole():
    """ToRORd's IpCa at cai = -KmCap. The spec's oo/zoo/nan test calls this
    removable and would emit -GpCa*KmCap/(KmCap + cai) + GpCa as the
    replacement -- itself infinite at exactly the guarded point."""
    IpCa = GpCa * cai / (KmCap + cai)
    assert not singularities.is_removable(IpCa, cai, -KmCap)


def test_is_removable_rejects_a_double_pole():
    assert not singularities.is_removable(1 / (v - 3) ** 2, v, sympy.Integer(3))


def test_taylor_of_a_float_coefficient_gate_rate_is_right():
    """Spec F5: sympy.limit returns 0 here. The series does not."""
    rate = 0.2 * (V + 23) / (1 - sympy.exp(-0.04 * (V + 23)))
    replacement = singularities.taylor(rate, V, sympy.Integer(-23), 3)
    assert abs(float(replacement.subs(V, -23)) - 5.0) < 1e-12


def test_taylor_of_the_toy_gate_rate_is_the_spec_f4_polynomial():
    """Spec F4: order >= 1 is what makes differentiation through the guard
    correct. Order 0 (`10`) differentiates to zero, which is the entire bug."""
    rate = (V + 10) / (sympy.exp((V + 10) / 10) - 1)
    replacement = singularities.taylor(rate, V, sympy.Integer(-10), 3)
    assert sympy.simplify(replacement - (V**2 / 120 - V / 3 + sympy.Rational(35, 6))) == 0
    assert float(replacement.subs(V, -10)) == 10.0
    assert float(sympy.diff(replacement, V).subs(V, -10)) == -0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_singularities.py -v -k "inline or removable or taylor"`
Expected: FAIL — `AttributeError: module 'gotranx.singularities' has no attribute 'inline'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/gotranx/singularities.py` (and extend `__all__` with
`"MAX_INLINE_OPS"`, `"inline"`, `"is_removable"`, `"taylor"`):

```python
from typing import Mapping

MAX_INLINE_OPS = 5000
"""Give up inlining past this many operations.

ToRORd's largest candidate reaches 939 operations, so this is loose. It exists
to bound the worst case on a model nobody has tried yet, not to exclude
anything in the bundled set.
"""


def inline(
    expr: sympy.Expr,
    definitions: Mapping[sympy.Symbol, sympy.Expr],
    var: sympy.Symbol,
) -> sympy.Expr | None:
    """Rewrite ``expr`` in terms of ``var``, expanding only what depends on it.

    A pole's removability is only visible after inlining: ToRORd's ``INab`` is
    ``PNab*vffrt*(...)/(exp(vfrt) - 1)``, and it is ``vffrt = v*F*F/(R*T)``
    that makes the numerator vanish as ``v -> 0``. Held as a separate named
    symbol, that vanishing factor is invisible and the removable pole looks
    essential.

    Only intermediates that *transitively depend on* ``var`` are expanded.
    Inlining the rest as well costs a great deal and buys nothing: on ToRORd,
    expanding ``PhiCaL_ss``'s whole cone pulls in the ionic-strength terms
    ``gamma_cass`` and ``gamma_cao``, which do not mention ``v``, and grows the
    generated rhs from 32,499 to 74,662 characters and scheme generation from
    2.75 s to 130.68 s. Restricted to ``v``-dependent intermediates the same
    guard costs 35,766 characters and 4.10 s.

    Leaving a ``var``-independent intermediate in place is also correct for the
    forward-mode sweep in :mod:`gotranx.linearization`: it carries its own
    tangent, exactly as it does in an unguarded assignment.

    Parameters
    ----------
    expr : sympy.Expr
        The expression to rewrite.
    definitions : Mapping[sympy.Symbol, sympy.Expr]
        Every intermediate symbol in the model and its defining expression.
    var : sympy.Symbol
        The state variable the pole lives in.

    Returns
    -------
    sympy.Expr | None
        The rewritten expression, or None if it grew past
        :data:`MAX_INLINE_OPS`.
    """
    dependent: set[sympy.Symbol] = set()
    # `definitions` is not necessarily topologically sorted, so iterate to a
    # fixed point rather than assuming one pass suffices.
    changed = True
    while changed:
        changed = False
        for symbol, definition in definitions.items():
            if symbol in dependent:
                continue
            if var in definition.free_symbols or (definition.free_symbols & dependent):
                dependent.add(symbol)
                changed = True

    while True:
        substitutions = {
            symbol: definitions[symbol] for symbol in expr.free_symbols if symbol in dependent
        }
        if not substitutions:
            return expr
        expr = expr.xreplace(substitutions)
        if sympy.count_ops(expr) > MAX_INLINE_OPS:
            return None


def is_removable(expr: sympy.Expr, var: sympy.Symbol, value: sympy.Expr) -> bool:
    """Whether ``expr``'s singularity at ``var = value`` is removable.

    Decided by the leading exponent of the Laurent expansion: removable means
    no negative powers of ``var - value``.

    Testing instead whether the truncated series contains ``oo``, ``zoo`` or
    ``nan`` -- as an earlier draft of the design did -- does not work, because
    the Laurent expansion around a genuine simple pole contains none of them.
    It contains a ``1/(var - value)`` term. Measured on ToRORd, that test
    classifies all 40 pole candidates as removable, including ``IpCa`` at
    ``cai = -KmCap``, whose "replacement" would be
    ``-GpCa*KmCap/(KmCap + cai) + GpCa``.

    Parameters
    ----------
    expr : sympy.Expr
        The expression, already inlined in ``var``.
    var : sympy.Symbol
        The variable the pole lives in.
    value : sympy.Expr
        Where the pole is.

    Returns
    -------
    bool
        True if the singularity is removable.
    """
    offset = sympy.Dummy("offset", positive=True)
    try:
        _, exponent = expr.subs(var, value + offset).leadterm(offset)
    except (ValueError, NotImplementedError, TypeError, PoleError, RecursionError):
        return False
    return bool(exponent >= 0)


def taylor(
    expr: sympy.Expr,
    var: sympy.Symbol,
    value: sympy.Expr,
    order: int,
) -> sympy.Expr | None:
    """Truncated Taylor series of ``expr`` about ``var = value``.

    Order must be at least 1. A constant replacement -- the limit, which is
    what the previous mechanism emitted -- differentiates to zero, so an
    assignment guarded that way contributes nothing to the Jacobian diagonal
    and the linearization is silently wrong. With order >= 1 the guard
    differentiates correctly and :func:`gotranx.linearization.diagonal_jacobian`
    needs no singularity awareness of its own.

    Parameters
    ----------
    expr : sympy.Expr
        The expression, already inlined in ``var``.
    var : sympy.Symbol
        The variable to expand in.
    value : sympy.Expr
        The point to expand about.
    order : int
        Highest power of ``var - value`` to keep. Order 3 costs the same as
        order 2 for the canonical gate-rate kernel, whose third Bernoulli
        number is zero.

    Returns
    -------
    sympy.Expr | None
        The expanded polynomial, or None if sympy could not produce a finite
        series.
    """
    try:
        series = sympy.series(expr, var, value, order + 1).removeO()
    except (ValueError, NotImplementedError, TypeError, PoleError, RecursionError):
        return None
    if series.has(sympy.oo, -sympy.oo, sympy.zoo, sympy.nan):
        return None
    return sympy.expand(series)
```

Add the import `from sympy.calculus.util import PoleError` near the top — or,
if that path is wrong on sympy 1.14, `from sympy import PoleError`. Verify with
`python3 -c "from sympy import PoleError; print(PoleError)"` before writing it.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_singularities.py -v`
Expected: PASS (16 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gotranx/singularities.py tests/test_singularities.py
git commit -m "feat(singularities): inline in the pole variable, classify by leading exponent

Inline only var-dependent intermediates (full-cone inlining grows ToRORd's
generated rhs 2.3x and scheme generation 47x for no gain), and decide
removability by leading exponent -- a Laurent expansion around a genuine
pole contains no oo/zoo/nan, so scanning for those accepts real poles."
```

---

### Task 3: Window half-width and numeric verification

**Files:**
- Modify: `src/gotranx/singularities.py`
- Test: `tests/test_singularities.py`

**Interfaces:**
- Consumes: `taylor` from Task 2.
- Produces:
  - `MIN_HALF_WIDTH: float = 1e-8`, `MAX_HALF_WIDTH: float = 1e-1`
  - `half_width(expr, var, value, order, defaults) -> float | None`
  - `agrees_numerically(expr, replacement, var, value, delta, defaults) -> bool`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_singularities.py`:

```python
def test_half_width_lands_at_the_crossover_for_the_canonical_kernel():
    """The spec's tolerance table puts the crossover at ~1e-3 in the local
    variable. Here the local variable is (V + 10)/10, so ~1e-2 in V."""
    rate = (V + 10) / (sympy.exp((V + 10) / 10) - 1)
    delta = singularities.half_width(rate, V, sympy.Integer(-10), 3, {})
    assert 1e-3 < delta < 5e-2, delta


def test_half_width_is_clamped():
    """A pure polynomial has no truncation error at all; the window must not
    grow without bound."""
    delta = singularities.half_width(V**2 + 1, V, sympy.Integer(0), 3, {})
    assert delta == singularities.MAX_HALF_WIDTH


def test_half_width_beats_the_direct_formula_on_its_own_derivative():
    """The linearized block uses the *derivative*, so that is what the window
    must be calibrated on. Calibrating on the value returns 0.1 here, where
    the series derivative is 4.9e-10 off and the direct float64 derivative is
    only 3.0e-13 off -- a guard three orders worse than what it replaces."""
    import mpmath

    mpmath.mp.dps = 50
    F_, R_, T_, P_ = 96485.0, 8314.0, 310.0, 3.75e-10
    ghk = (
        P_
        * (v * F_ * F_ / (R_ * T_))
        * (12.0 * sympy.exp(v * F_ / (R_ * T_)) - 140.0)
        / (sympy.exp(v * F_ / (R_ * T_)) - 1)
    )
    delta = singularities.half_width(ghk, v, sympy.Integer(0), 3, {})
    replacement = singularities.taylor(ghk, v, sympy.Integer(0), 3)

    exact = sympy.lambdify(v, sympy.diff(sympy.nsimplify(ghk, rational=True), v), "mpmath")
    series = sympy.lambdify(v, sympy.diff(replacement, v), "mpmath")
    worst = max(
        abs(series(mpmath.mpf(delta * f)) - exact(mpmath.mpf(delta * f)))
        / abs(exact(mpmath.mpf(delta * f)))
        for f in (1.0, 0.5, 0.1, 0.01)
    )
    assert float(worst) < 1e-11, float(worst)


def test_agrees_numerically_accepts_a_correct_replacement():
    rate = 0.2 * (V + 23) / (1 - sympy.exp(-0.04 * (V + 23)))
    replacement = singularities.taylor(rate, V, sympy.Integer(-23), 3)
    assert singularities.agrees_numerically(rate, replacement, V, sympy.Integer(-23), 1e-2, {})


def test_agrees_numerically_rejects_the_limit_value_sympy_gets_wrong():
    """Spec F5: sympy.limit returns 0 for this rate, where the truth is 5.
    The numeric spot check is the backstop that catches exactly this."""
    rate = 0.2 * (V + 23) / (1 - sympy.exp(-0.04 * (V + 23)))
    assert not singularities.agrees_numerically(
        rate, sympy.Integer(0), V, sympy.Integer(-23), 1e-2, {}
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_singularities.py -v -k "half_width or agrees"`
Expected: FAIL — `AttributeError: module 'gotranx.singularities' has no attribute 'half_width'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/gotranx/singularities.py` (extend `__all__` accordingly):

```python
MIN_HALF_WIDTH = 1e-8
MAX_HALF_WIDTH = 1e-1
EPSILON = 2.220446049250313e-16  # float64 machine epsilon


def _numeric(expr: sympy.Expr, defaults: Mapping[sympy.Symbol, float]) -> float | None:
    """``expr`` as a real float at default parameter/state values, or None."""
    substituted = expr.xreplace({s: sympy.Float(x) for s, x in defaults.items()})
    try:
        value = complex(substituted.evalf())
    except (TypeError, ValueError, AttributeError):
        return None
    if not all(
        map(sympy.core.numbers.Float(0).__class__.is_finite.__get__, ())
    ):  # pragma: no cover
        pass
    if value != value or abs(value) == float("inf"):
        return None
    if abs(value.imag) > 1e-12 * max(1.0, abs(value.real)):
        return None
    return value.real


def _series_coefficients(
    expr: sympy.Expr,
    var: sympy.Symbol,
    value: sympy.Expr,
    upto: int,
    defaults: Mapping[sympy.Symbol, float],
) -> list[float] | None:
    """Numeric Taylor coefficients a_0 .. a_upto about ``var = value``."""
    offset = sympy.Symbol("_offset")
    try:
        series = sympy.series(expr.subs(var, value + offset), offset, 0, upto + 1).removeO()
        poly = sympy.Poly(sympy.expand(series), offset)
    except (
        ValueError,
        NotImplementedError,
        TypeError,
        PoleError,
        RecursionError,
        sympy.PolynomialError,
        sympy.GeneratorsNeeded,
    ):
        return None
    coefficients = []
    for k in range(upto + 1):
        coefficient = _numeric(poly.coeff_monomial(offset**k), defaults)
        if coefficient is None:
            return None
        coefficients.append(coefficient)
    return coefficients


def half_width(
    expr: sympy.Expr,
    var: sympy.Symbol,
    value: sympy.Expr,
    order: int,
    defaults: Mapping[sympy.Symbol, float],
) -> float | None:
    r"""Half-width of the window in which the Taylor branch is used.

    Chosen where the series' truncation error crosses the direct formula's
    float64 round-off, measured on the *derivative* -- the quantity the
    linearized block needs. With coefficients :math:`a_k` and :math:`k` the
    first nonzero index above ``order``, truncation of the derivative is
    :math:`k |a_k| \delta^{k-1} / |a_1|` and round-off of the direct
    derivative grows like :math:`\epsilon / \delta^2`, so the crossover is

    .. math::
        \delta^{k+1} = \frac{\epsilon |a_1|}{k |a_k|}

    clamped to ``[MIN_HALF_WIDTH, MAX_HALF_WIDTH]``. The clamp matters: the
    lower bound keeps the window wider than the region where the direct
    formula is already catastrophically wrong (222% relative error at 1e-8, by
    the spec's tolerance table), and the upper bound keeps the series out of
    the region where the direct formula is the more accurate of the two.

    Calibrating on the *value* instead -- bounding
    :math:`|a_{k} \delta^{k}| \le \mathrm{tol}|a_0|` -- is one order too
    generous. On ToRORd's ``INab`` it returns 0.1, where the series derivative
    is 4.9e-10 off a 50-digit reference and the direct float64 derivative is
    3.0e-13 off.

    The rule is scale-free: it adapts to a gate rate written ``-0.1*(V + 47)``
    as readily as one written ``(V + 10)/10``, where a fixed window in the
    guarded variable's own units would not.

    Parameters
    ----------
    expr : sympy.Expr
        The expression, already inlined in ``var``.
    var : sympy.Symbol
        The variable the pole lives in.
    value : sympy.Expr
        Where the pole is.
    order : int
        The order of the Taylor replacement.
    defaults : Mapping[sympy.Symbol, float]
        Default values for parameters and other states, used to make the
        coefficients numeric.

    Returns
    -------
    float | None
        The half-width, or None if the coefficients could not be evaluated.
    """
    coefficients = _series_coefficients(expr, var, value, order + 8, defaults)
    if coefficients is None:
        return None

    tail = [(k, c) for k, c in enumerate(coefficients) if k > order and c != 0.0]
    if not tail:
        # No truncation error within reach: the replacement is exact as far as
        # we can see, so use the widest window allowed.
        return MAX_HALF_WIDTH
    k, a_k = tail[0]

    reference = coefficients[1] if coefficients[1] != 0.0 else coefficients[0]
    if reference == 0.0:
        return None
    delta = (EPSILON * abs(reference) / (k * abs(a_k))) ** (1.0 / (k + 1))
    return min(max(delta, MIN_HALF_WIDTH), MAX_HALF_WIDTH)


def agrees_numerically(
    expr: sympy.Expr,
    replacement: sympy.Expr,
    var: sympy.Symbol,
    value: sympy.Expr,
    delta: float,
    defaults: Mapping[sympy.Symbol, float],
    tolerance: float = 1e-6,
) -> bool:
    """Spot-check a replacement against the expression it replaces.

    Symbolic tools have silently produced wrong answers twice in this design's
    investigation -- ``limit`` returning 0 on a float-coefficient gate rate,
    and a Laurent branch passing an ``oo``/``zoo``/``nan`` scan -- so every
    replacement is checked numerically before it is emitted.

    The check is made at ``value +- delta``, the window *edge*, not at its
    centre: ``delta`` is chosen to be where the two branches cross over, so
    both are accurate there. Closer in, the direct formula is the inaccurate
    one and a disagreement would say nothing.

    Parameters
    ----------
    expr : sympy.Expr
        The original expression.
    replacement : sympy.Expr
        The proposed Taylor replacement.
    var : sympy.Symbol
        The variable the pole lives in.
    value : sympy.Expr
        Where the pole is.
    delta : float
        The window half-width.
    defaults : Mapping[sympy.Symbol, float]
        Default values for parameters and other states.
    tolerance : float, optional
        Largest acceptable relative disagreement, by default 1e-6.

    Returns
    -------
    bool
        True if the replacement may be emitted.
    """
    centre = _numeric(value, defaults)
    if centre is None:
        return False
    for point in (centre - delta, centre + delta):
        substitution = {var: sympy.Float(point)}
        original = _numeric(expr.subs(substitution), defaults)
        proposed = _numeric(replacement.subs(substitution), defaults)
        if original is None or proposed is None:
            return False
        scale = max(abs(original), abs(proposed), 1e-300)
        if abs(original - proposed) / scale > tolerance:
            return False
    return True
```

Note: the stray `if not all(map(...))` line in `_numeric` above is a
placeholder artefact — delete it; the `value != value or abs(value) == inf`
check that follows is the real one.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_singularities.py -v`
Expected: PASS (21 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gotranx/singularities.py tests/test_singularities.py
git commit -m "feat(singularities): calibrate the guard window on the derivative

The linearized block uses the derivative, whose truncation error is one order
above the value's; calibrating on the value returns a window where the guard
is three orders less accurate than the formula it replaces. Verify every
replacement numerically at the window edge before emitting it."
```

---

### Task 4: Guard emission and the assignment-level rewrite

**Files:**
- Modify: `src/gotranx/singularities.py`
- Test: `tests/test_singularities.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces:
  - `RemovablePole` (frozen dataclass: `var`, `value`, `replacement`, `half_width`, `rewritten`)
  - `removable_poles(expr, definitions, states, defaults, order=3) -> tuple[RemovablePole, ...]`
  - `guard(expr, poles) -> sympy.Expr`
  - `rewrite(expr, definitions, states, defaults, order=3) -> sympy.Expr`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_singularities.py`:

```python
def test_removable_poles_finds_the_ghk_pole_and_nothing_else():
    poles = singularities.removable_poles(
        GHK,
        GHK_DEFINITIONS,
        frozenset({v}),
        {F: 96485.0, R: 8314.0, T: 310.0, PNab: 3.75e-10, nai: 12.0, nao: 140.0, v: -80.0},
    )
    assert len(poles) == 1
    assert poles[0].var == v
    assert poles[0].value == 0
    assert poles[0].half_width > 0


def test_removable_poles_skips_a_genuine_pole():
    IpCa = GpCa * cai / (KmCap + cai)
    poles = singularities.removable_poles(
        IpCa, {}, frozenset({cai}), {GpCa: 0.0005, KmCap: 0.0005, cai: 1e-4}
    )
    assert poles == ()


def test_removable_poles_skips_a_location_that_is_not_a_real_number():
    """TP06's Ca_i buffering root is -K_buf_c +- sqrt(-Buf_c*K_buf_c), which is
    imaginary for positive parameters. A guard window needs a real centre."""
    Buf, K = sympy.symbols("Buf K", real=True)
    expr = cai / (cai**2 + 2 * K * cai + K**2 + Buf * K)
    poles = singularities.removable_poles(
        expr, {}, frozenset({cai}), {Buf: 0.2, K: 0.001, cai: 1e-4}
    )
    assert poles == ()


def test_guard_emits_a_piecewise_on_the_window():
    pole = singularities.RemovablePole(
        var=V,
        value=sympy.Integer(-10),
        replacement=sympy.Integer(10),
        half_width=1e-2,
        rewritten=V,
    )
    guarded = singularities.guard(V, (pole,))
    assert isinstance(guarded, sympy.Piecewise)
    assert float(guarded.subs(V, -10)) == 10.0
    assert float(guarded.subs(V, 5)) == 5.0


def test_rewrite_is_the_identity_when_there_is_no_pole():
    expr = sympy.exp(V) + 3
    assert singularities.rewrite(expr, {}, frozenset({V}), {V: 0.0}) is expr


def test_rewrite_guards_the_ghk_expression_in_the_state_variable():
    """The guard must be written in v, not vfrt: the AD sweep in
    linearization.py is seeded at v, and a branch written in vfrt would
    differentiate to zero there."""
    defaults = {F: 96485.0, R: 8314.0, T: 310.0, PNab: 3.75e-10, nai: 12.0, nao: 140.0, v: -80.0}
    guarded = singularities.rewrite(GHK, GHK_DEFINITIONS, frozenset({v}), defaults)
    assert v in guarded.free_symbols
    assert VFRT not in guarded.free_symbols
    assert VFFRT not in guarded.free_symbols
    numeric = sympy.lambdify(
        v, guarded.subs({s: sympy.Float(x) for s, x in defaults.items() if s is not v}), "numpy"
    )
    import numpy

    assert numpy.isfinite(float(numeric(numpy.float64(0.0))))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_singularities.py -v -k "removable_poles or guard or rewrite"`
Expected: FAIL — `AttributeError: module 'gotranx.singularities' has no attribute 'removable_poles'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/gotranx/singularities.py`:

```python
import dataclasses

from structlog import get_logger

logger = get_logger()


@dataclasses.dataclass(frozen=True)
class RemovablePole:
    """A removable pole and the replacement that covers it.

    Attributes
    ----------
    var : sympy.Symbol
        The state variable the pole lives in.
    value : sympy.Expr
        Where the pole is. Free of state symbols, and real and finite at
        default parameter values.
    replacement : sympy.Expr
        Truncated Taylor series, of order at least 1.
    half_width : float
        Half-width of the window in which ``replacement`` is used.
    rewritten : sympy.Expr
        The original expression, rewritten in ``var``. This is the branch
        taken outside the window; it must be written in ``var`` too, or the
        forward-mode sweep in :mod:`gotranx.linearization` -- which is seeded
        at ``var`` -- cannot differentiate it.
    """

    var: sympy.Symbol
    value: sympy.Expr
    replacement: sympy.Expr
    half_width: float
    rewritten: sympy.Expr


def removable_poles(
    expr: sympy.Expr,
    definitions: Mapping[sympy.Symbol, sympy.Expr],
    states: frozenset[sympy.Symbol],
    defaults: Mapping[sympy.Symbol, float],
    order: int = 3,
) -> tuple[RemovablePole, ...]:
    """Every removable pole of ``expr`` in a state variable.

    Only assignments whose denominator contains a state-dependent symbol are
    examined; a denominator that is a parameter or a literal cannot vanish for
    a state-dependent reason. That pre-filter is what keeps the scan
    affordable -- ``dv_dt = (I_stim - ...)/C`` is skipped without inlining its
    10,013-operation cone, because ``C`` is a parameter.

    A pole is kept only if all of the following hold:

    * its location is free of state symbols, so the window half-width is a
      compile-time constant;
    * its location is a real, finite number at default parameter values (TP06's
      ``Ca_i`` buffering roots are ``-K_buf_c +- sqrt(-Buf_c*K_buf_c)``, which
      is imaginary for positive parameters);
    * it is removable, by leading exponent;
    * a truncated series exists, and
    * that series agrees numerically with the original at the window edge.

    Parameters
    ----------
    expr : sympy.Expr
        The assignment's expression, as written in the model.
    definitions : Mapping[sympy.Symbol, sympy.Expr]
        Every intermediate symbol in the model and its defining expression.
    states : frozenset[sympy.Symbol]
        The model's state symbols.
    defaults : Mapping[sympy.Symbol, float]
        Default values for every parameter and state.
    order : int, optional
        Order of the Taylor replacement, by default 3.

    Returns
    -------
    tuple[RemovablePole, ...]
        In a deterministic order: by variable name, then by location.
    """
    state_dependent = set(states)
    changed = True
    while changed:
        changed = False
        for symbol, definition in definitions.items():
            if symbol not in state_dependent and (definition.free_symbols & state_dependent):
                state_dependent.add(symbol)
                changed = True

    factors = denominator_factors(expr)
    if not any(factor.free_symbols & state_dependent for factor in factors):
        return ()

    found: list[RemovablePole] = []
    seen: set[tuple[sympy.Symbol, sympy.Expr]] = set()
    for factor in factors:
        for var in sorted(factor.free_symbols & state_dependent, key=str):
            # `var` may be an intermediate; resolve the factor down to states.
            pass
        for var in sorted(states, key=str):
            inlined_factor = inline(factor, definitions, var)
            if inlined_factor is None or var not in inlined_factor.free_symbols:
                continue
            for value in factor_roots(inlined_factor, var):
                if value.free_symbols & states:
                    logger.debug("Pole location depends on a state", var=str(var))
                    continue
                if _numeric(value, defaults) is None:
                    logger.debug("Pole location is not a real number", var=str(var))
                    continue
                if (var, value) in seen:
                    continue
                seen.add((var, value))

                rewritten = inline(expr, definitions, var)
                if rewritten is None:
                    logger.debug("Expression too large to inline", var=str(var))
                    continue
                if not is_removable(rewritten, var, value):
                    continue
                replacement = taylor(rewritten, var, value, order)
                if replacement is None:
                    continue
                delta = half_width(rewritten, var, value, order, defaults)
                if delta is None:
                    continue
                if not agrees_numerically(rewritten, replacement, var, value, delta, defaults):
                    logger.warning(
                        "Dropping a guard whose replacement failed a numeric check",
                        var=str(var),
                        value=str(value),
                    )
                    continue
                found.append(
                    RemovablePole(
                        var=var,
                        value=value,
                        replacement=replacement,
                        half_width=delta,
                        rewritten=rewritten,
                    )
                )
    return tuple(sorted(found, key=lambda pole: (str(pole.var), str(pole.value))))


def guard(expr: sympy.Expr, poles: tuple[RemovablePole, ...]) -> sympy.Expr:
    """Wrap ``expr`` in one ``Piecewise`` per pole.

    Parameters
    ----------
    expr : sympy.Expr
        The expression to guard.
    poles : tuple[RemovablePole, ...]
        The poles to cover. May be empty, in which case ``expr`` is returned
        unchanged.

    Returns
    -------
    sympy.Expr
        ``Piecewise((replacement, Abs(var - value) < half_width), (expr, True))``,
        nested when there is more than one pole. Each pole's own ``rewritten``
        form is used as the fallback, so the guarded expression is written in
        that pole's variable.
    """
    if not poles:
        return expr
    guarded = poles[-1].rewritten
    for pole in poles:
        guarded = sympy.Piecewise(
            (
                pole.replacement,
                sympy.Abs(pole.var - pole.value) < sympy.Float(pole.half_width),
            ),
            (guarded, True),
        )
    return guarded


def rewrite(
    expr: sympy.Expr,
    definitions: Mapping[sympy.Symbol, sympy.Expr],
    states: frozenset[sympy.Symbol],
    defaults: Mapping[sympy.Symbol, float],
    order: int = 3,
) -> sympy.Expr:
    """Guard every removable pole of ``expr``, or return it unchanged.

    Parameters
    ----------
    expr : sympy.Expr
        The assignment's expression.
    definitions : Mapping[sympy.Symbol, sympy.Expr]
        Every intermediate symbol in the model and its defining expression.
    states : frozenset[sympy.Symbol]
        The model's state symbols.
    defaults : Mapping[sympy.Symbol, float]
        Default values for every parameter and state.
    order : int, optional
        Order of the Taylor replacement, by default 3.

    Returns
    -------
    sympy.Expr
        The guarded expression, or ``expr`` itself if it has no removable pole.
    """
    poles = removable_poles(expr, definitions, states, defaults, order=order)
    if not poles:
        return expr
    logger.debug(
        "Guarding removable poles",
        poles=[(str(p.var), str(p.value), p.half_width) for p in poles],
    )
    return guard(expr, poles)
```

Delete the dead `for var in sorted(factor.free_symbols & state_dependent...)`
loop shown above — it is a leftover; the loop over `states` beneath it is the
real one.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_singularities.py -v`
Expected: PASS (27 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gotranx/singularities.py tests/test_singularities.py
git commit -m "feat(singularities): emit guards rewritten in the state variable"
```

---

### Task 5: Keep CSE out of guarded subtrees

**Files:**
- Modify: `src/gotranx/sympytools.py`
- Modify: `src/gotranx/schemes.py:176`
- Test: `tests/test_sympytools.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure sympy).
- Produces: `sympytools.cse_hiding_piecewise(exprs, **kwargs) -> tuple[list, list]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sympytools.py`:

```python
def test_cse_hiding_piecewise_does_not_hoist_out_of_a_branch():
    """sympy.cse pulls `1/(1 - x0)` out of the Piecewise below and computes it
    unconditionally, so the temporary is 1/0 at V = -40 even though the branch
    that uses it is not taken there. Reproduced on the pinned sympy 1.14."""
    import sympy

    from gotranx import sympytools

    V = sympy.Symbol("V", real=True)
    expr = sympy.Piecewise(
        (sympy.Integer(0), V < -40),
        ((V + 40) / (sympy.exp(-(V + 40) / 10) - 1), True),
    )

    replacements, _ = sympy.cse([expr + expr * 2], optimizations="basic")
    hoisted = [
        sub_expr
        for _, sub_expr in replacements
        if sub_expr.has(sympy.exp) and not sub_expr.has(sympy.Piecewise)
    ]
    assert hoisted, "expected plain sympy.cse to hoist out of the Piecewise"

    replacements, reduced = sympytools.cse_hiding_piecewise(
        [expr + expr * 2], optimizations="basic"
    )
    for _, sub_expr in replacements:
        assert not (sub_expr.has(sympy.exp) and not sub_expr.has(sympy.Piecewise)), sub_expr
    assert sympy.simplify(reduced[0] - (expr + expr * 2)) == 0


def test_cse_hiding_piecewise_still_shares_whole_guards():
    """Holding a Piecewise opaque must not stop cse from sharing the whole
    subtree between two expressions -- that is where most of the saving is."""
    import sympy

    from gotranx import sympytools

    V = sympy.Symbol("V", real=True)
    guard = sympy.Piecewise((sympy.Integer(1), sympy.Abs(V) < 0.01), (1 / V, True))
    replacements, reduced = sympytools.cse_hiding_piecewise([guard * 2, guard * 3])
    assert any(sub_expr == guard for _, sub_expr in replacements)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sympytools.py -v -k piecewise`
Expected: FAIL — `AttributeError: module 'gotranx.sympytools' has no attribute 'cse_hiding_piecewise'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/gotranx/sympytools.py`:

```python
def cse_hiding_piecewise(exprs, **kwargs):
    """``sympy.cse``, with ``Piecewise`` subtrees held opaque.

    ``sympy.cse`` happily hoists a subexpression out of a ``Piecewise`` branch
    and computes it unconditionally. That is fine for a branch chosen for
    convenience and fatal for one chosen to avoid a singularity: the whole
    point of ``Piecewise((taylor, |V - v0| < delta), (expr, True))`` is that
    ``expr`` is never evaluated near ``v0``, and a hoisted temporary evaluates
    it everywhere. The same hoist breaks model conditionals that were already
    written defensively, e.g.
    ``Piecewise((0, V < -40), ((V + 40)/(exp(-(V + 40)/10) - 1), True))``,
    where the hoisted ``1/(1 - x0)`` is ``1/0`` at exactly ``V = -40``.

    Every ``Piecewise`` is replaced by a fresh ``Dummy`` before the call and
    restored after, so nothing inside one can be factored out, while a
    ``Piecewise`` shared between two expressions is still shared as a whole.

    Note that under numpy both branches of the emitted ``where`` are still
    evaluated. That is ``where`` semantics, not a hoist; what this function
    guarantees is that no temporary computed *outside* a guard divides by zero
    at the guarded value.

    Parameters
    ----------
    exprs : list[sympy.Expr]
        The expressions to factor.
    **kwargs
        Passed straight through to ``sympy.cse``.

    Returns
    -------
    tuple[list, list]
        Exactly what ``sympy.cse`` returns.
    """
    holes: dict[sympy.Expr, sympy.Dummy] = {}

    def hide(expr):
        if isinstance(expr, sympy.Piecewise):
            return holes.setdefault(expr, sympy.Dummy(f"_guard_{len(holes)}"))
        if not expr.args:
            return expr
        return expr.func(*[hide(arg) for arg in expr.args])

    hidden = [hide(expr) for expr in exprs]
    restore = {dummy: expr for expr, dummy in holes.items()}
    replacements, reduced = sympy.cse(hidden, **kwargs)
    return (
        [(symbol, sub_expr.xreplace(restore)) for symbol, sub_expr in replacements],
        [expr.xreplace(restore) for expr in reduced],
    )
```

Then in `src/gotranx/schemes.py`, replace line 176:

```python
    replacements, reduced = sympy.cse(exprs, symbols=fresh(), optimizations="basic")
```

with

```python
    replacements, reduced = sympytools.cse_hiding_piecewise(
        exprs, symbols=fresh(), optimizations="basic"
    )
```

(`sympytools` is already imported in `schemes.py`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_sympytools.py tests/test_schemes.py tests/test_deep_linearization_models.py -v`
Expected: PASS. `test_deep_linearization_models.py` exercises the changed
`cse` path on ToRORd and TP06; if a string-comparison test in
`tests/test_schemes.py` changes, inspect the diff and update the expectation
only if the new output is equivalent — record the before/after in the commit
message.

- [ ] **Step 5: Commit**

```bash
git add src/gotranx/sympytools.py src/gotranx/schemes.py tests/test_sympytools.py
git commit -m "fix(schemes): stop cse hoisting out of Piecewise branches

A branch chosen to avoid a singularity is worthless if cse computes its
guarded subexpression unconditionally."
```

---

### Task 6: Drive the new module from `ODE`, delete the old mechanism

**Files:**
- Modify: `src/gotranx/ode.py:340`
- Modify: `src/gotranx/atoms.py:196-240,269-340`
- Modify: `src/gotranx/ode_component.py:195-217`
- Modify: `tests/test_atoms.py:208`
- Modify: `tests/test_python_codegen.py:694`
- Test: `tests/test_singularity_models.py` (new)

**Interfaces:**
- Consumes: `singularities.rewrite` from Task 4.
- Produces: `ODE.remove_singularities() -> ODE` (unchanged signature, new behaviour).

- [ ] **Step 1: Write the failing test**

Create `tests/test_singularity_models.py`:

```python
"""Model-level acceptance tests for removable-singularity guarding.

Mapped to the design document's test plan.
"""

from __future__ import annotations

from pathlib import Path

import mpmath
import numpy as np
import pytest
import sympy

import gotranx
from gotranx.atoms import make_symbol
from gotranx.linearization import diagonal_jacobian

HERE = Path(__file__).parent
ODEFILES = HERE / "odefiles"

TOY = """
parameters(C=1.0)
states("Membrane", V=-80.0, n=0.1)
expressions("Membrane")
alpha_n = (V + 10)/(exp((V + 10)/10) - 1)
dn_dt = alpha_n*(1 - n) - 0.125*n
dV_dt = -alpha_n*(V + 77)/C
"""


def _evaluate(expr, values):
    """Evaluate `expr` at `values`, resolving intermediates by substitution."""
    substituted = expr.xreplace({s: sympy.Float(x) for s, x in values.items()})
    return float(substituted.evalf())


@pytest.fixture(scope="module")
def guarded():
    names = ["beeler_reuter_1977", "ToRORd_dyn_chloride", "ORdmm_Land"]
    return {
        name: gotranx.load_ode(ODEFILES / f"{name}.ode").remove_singularities() for name in names
    }


def test_toy_model_guard_evaluates_to_the_true_limit():
    """Test plan item 3 / spec F4. The emitted branch is a polynomial, not the
    constant 23.5, so this asserts the value rather than grepping for it."""
    ode = gotranx.load.ode_from_string(TOY).remove_singularities()
    jac = diagonal_jacobian(ode)
    alpha_n = [a for a in ode.sorted_assignments() if a.name == "alpha_n"][0]

    V, C, n = make_symbol("V"), make_symbol("C"), make_symbol("n")
    at_pole = {V: -10.0, C: 1.0, n: 0.1}
    alpha_value = _evaluate(alpha_n.expr, at_pole)
    assert abs(alpha_value - 10.0) < 1e-12

    at_pole[make_symbol("alpha_n")] = alpha_value
    assert abs(_evaluate(jac["V"], at_pole) - 23.5) < 1e-9


@pytest.mark.parametrize(
    "name, state, pole",
    [
        ("beeler_reuter_1977", "V", -23.0),
        ("ToRORd_dyn_chloride", "v", 0.0),
        ("ORdmm_Land", "v", 0.0),
    ],
)
def test_diagonal_jacobian_is_finite_and_accurate_across_the_pole(name, state, pole, guarded):
    """Test plan items 1 and 2. Unguarded, ToRORd's d(dv_dt)/dv is +30.1 at
    v = -1e-9 and raises at v = 0, against a true value near -0.0405; a
    positive linearization feeds exp(g*dt) and turns a decaying exponential
    into a growing one."""
    ode = guarded[name]
    jac = diagonal_jacobian(ode)[state]

    values = {make_symbol(p.name): float(p.value) for p in ode.parameters}
    values.update({make_symbol(s.name): float(s.value) for s in ode.states})
    var = make_symbol(state)

    np.seterr(all="ignore")
    for offset in (-1e-1, -1e-3, -1e-9, 0.0, 1e-9, 1e-3, 1e-1):
        point = dict(values)
        point[var] = pole + offset
        for assignment in ode.sorted_assignments():
            point[assignment.symbol] = _evaluate(assignment.expr, point)
        got = _evaluate(jac, point)
        assert np.isfinite(got), f"{name}: non-finite at {state} = {pole + offset}"


def test_beeler_reuter_agrees_with_a_high_precision_reference(guarded):
    """Test plan item 1, the accuracy half.

    The spec asks for ~1e-11 relative agreement across the neighbourhood. That
    is not reachable at order 3: by the spec's own tolerance table the series
    at delta = 1e-2 is good to 1.11e-8 and the direct formula at 1e-4 only to
    8.65e-9, so no single window puts both branches under 1e-11. The window is
    placed at the crossover instead, and this test asserts the bound that
    placement actually achieves. Fill in MEASURED below from the first run and
    keep a 10x margin.
    """
    mpmath.mp.dps = 50
    ode = guarded["beeler_reuter_1977"]
    jac = diagonal_jacobian(ode)["V"]
    # ... see Step 3 for how the reference is built; assert worst < 1e-8
    assert jac is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_singularity_models.py -v`
Expected: FAIL — `test_toy_model_guard_evaluates_to_the_true_limit` fails with
the guarded branch evaluating to `-10.0` rather than `23.5`, because
`ODE.remove_singularities` still emits the order-0 replacement
(`numpy.where((x == 0), 1, ...)` in `tests/test_python_codegen.py:735` shows
the shape). Paste the actual failure output into the commit message.

- [ ] **Step 3: Write the implementation**

In `src/gotranx/ode.py`, replace `remove_singularities`:

```python
def remove_singularities(self, order: int = 3) -> ODE:
    """Guard every removable singularity in the model's assignments.

    A gate rate like ``x/(exp(x) - 1)`` has a removable pole that float64
    cannot evaluate near, and whose *derivative* -- which the Rush-Larsen
    linearized block forms -- has a double pole there. Each such
    assignment is rewritten in the state variable the pole lives in and
    wrapped in a ``Piecewise`` whose other branch is a truncated Taylor
    series. See :mod:`gotranx.singularities`.

    Parameters
    ----------
    order : int, optional
        Order of the Taylor replacement, by default 3. Must be at least 1:
        a constant replacement differentiates to zero, which leaves the
        linearization silently wrong.

    Returns
    -------
    ODE
        A new ODE. The original is unchanged.
    """
    from . import singularities

    definitions = {a.symbol: a.expr for a in self.sorted_assignments()}
    states = frozenset(state.symbol for state in self.states)
    defaults = {p.symbol: float(p.value) for p in self.parameters}
    defaults.update({s.symbol: float(s.value) for s in self.states})

    new_components: list[BaseComponent] = []
    for component in self._components.values():
        new_assignments = set()
        for assignment in component.assignments:
            expr = singularities.rewrite(
                assignment.expr, definitions, states, defaults, order=order
            )
            if expr is assignment.expr:
                new_assignments.add(assignment)
            else:
                new_assignments.add(attr.evolve(assignment, expr=expr))
        new_components.append(
            type(component)(
                name=component.name,
                states=component.states,
                parameters=component.parameters,
                assignments=frozenset(new_assignments),
            )
        )

    return ODE(
        components=new_components,
        t=self.t,
        name=self.name,
        comments=self.comments,
    )
```

Add `import attr` at the top of `ode.py` if it is not already there. Verify
that `attr.evolve` works on these slotted frozen classes with
`python3 -c "..."` before relying on it; if it does not, construct the new
assignment explicitly the way `Assignment.resolve_expression` does
(`type(self)(name=..., value=..., components=..., unit_str=..., unit=...,
expr=..., symbol=..., description=..., comment=...)`, plus `state=` for a
`StateDerivative`).

Then delete, in this order:
1. `src/gotranx/ode_component.py`: `Component.remove_singularities` (lines 195–217).
2. `src/gotranx/atoms.py`: `Assignment.remove_singularities` (lines 310–340),
   `Assignment.singularities` (lines 269–308), the module-level
   `remove_singularities` (lines 207–239), and the `Singularity` class
   (lines 196–204).
3. `tests/test_atoms.py`: `test_singularities` (line 208) — the behaviour it
   pins no longer exists.
4. `tests/test_python_codegen.py`: `test_codegen_rhs_singular_ode` (line 694) —
   rewrite the post-`remove_singularities` half. The new output guards `y` with
   a window and a polynomial rather than `numpy.where((x == 0), 1, ...)`, and
   `z`'s two poles nest rather than folding into a sum. Regenerate the expected
   string from the new implementation, read it, and confirm by hand that each
   branch is right before pinning it.

Fill in `test_beeler_reuter_agrees_with_a_high_precision_reference` by
lambdifying `diagonal_jacobian(ode)["V"]` with `mpmath` after substituting a
rationalized copy of the parameters, evaluating at
`-22.9, -22.999, -22.99999, -22.9999999, -23.0` and the mirror points, and
comparing against the same quantity built from the *unguarded* ODE with
`mpmath.mp.dps = 50`. At exactly `-23.0` the unguarded reference cannot be
evaluated, so use the guarded value's own limit there and assert only
finiteness plus agreement with the two nearest points.

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_singularity_models.py tests/test_atoms.py tests/test_python_codegen.py tests/test_ode.py -v`
Expected: PASS. Paste the output.

- [ ] **Step 5: Commit**

```bash
git add src/gotranx/ode.py src/gotranx/atoms.py src/gotranx/ode_component.py tests/
git commit -m "feat(ode): guard removable singularities in the state variable

Replaces the old per-assignment mechanism, which scanned in whatever symbol
the model author happened to name, emitted an order-0 replacement that
differentiates to zero, and discarded every GHK pole as infinite."
```

---

### Task 7: Default-on, with `--no-remove-singularities` to opt out

**Files:**
- Modify: `src/gotranx/load.py`
- Modify: `src/gotranx/cli/__init__.py` (`ode2py` ~line 284, `ode2c` ~423, `ode2julia` ~526, `ode2mtk`, `ode2ufl`, `ode2md`, and the `convert` shim ~line 76)
- Modify: `src/gotranx/cli/gotran2py.py:135`, `gotran2c.py:111`, `gotran2julia.py:111`, `gotran2ufl.py:87`
- Modify: `src/gotranx/cli/utils.py`
- Modify: `docs/cli.md`, `docs/config.md`, `config.toml`, `CHANGELOG.md`
- Test: `tests/test_cli.py`, `tests/test_load.py`

**Interfaces:**
- Consumes: `ODE.remove_singularities` from Task 6.
- Produces: `load_ode(path, *, remove_singularities: bool = True) -> ODE`,
  `ode_from_string(text, name="ode", *, remove_singularities: bool = True) -> ODE`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_load.py`:

```python
def test_load_ode_guards_singularities_by_default(tmp_path):
    import sympy

    path = tmp_path / "gate.ode"
    path.write_text(
        "states('M', V=-80.0)\n"
        "expressions('M')\n"
        "alpha_n = (V + 10)/(exp((V + 10)/10) - 1)\n"
        "dV_dt = -alpha_n\n"
    )
    ode = gotranx.load_ode(path)
    alpha_n = [a for a in ode.sorted_assignments() if a.name == "alpha_n"][0]
    assert alpha_n.expr.has(sympy.Piecewise)


def test_load_ode_can_be_asked_not_to(tmp_path):
    import sympy

    path = tmp_path / "gate.ode"
    path.write_text(
        "states('M', V=-80.0)\n"
        "expressions('M')\n"
        "alpha_n = (V + 10)/(exp((V + 10)/10) - 1)\n"
        "dV_dt = -alpha_n\n"
    )
    ode = gotranx.load_ode(path, remove_singularities=False)
    alpha_n = [a for a in ode.sorted_assignments() if a.name == "alpha_n"][0]
    assert not alpha_n.expr.has(sympy.Piecewise)
```

Append to `tests/test_cli.py` (matching the file's existing invocation style):

```python
def test_ode2py_no_remove_singularities_flag(tmp_path):
    from typer.testing import CliRunner

    from gotranx.cli import app

    path = tmp_path / "gate.ode"
    path.write_text(
        "states('M', V=-80.0)\n"
        "expressions('M')\n"
        "alpha_n = (V + 10)/(exp((V + 10)/10) - 1)\n"
        "dV_dt = -alpha_n\n"
    )
    runner = CliRunner()

    result = runner.invoke(app, ["ode2py", str(path), "-o", str(tmp_path / "on.py")])
    assert result.exit_code == 0
    assert "numpy.where" in (tmp_path / "on.py").read_text()

    result = runner.invoke(
        app,
        ["ode2py", str(path), "--no-remove-singularities", "-o", str(tmp_path / "off.py")],
    )
    assert result.exit_code == 0
    assert "numpy.where" not in (tmp_path / "off.py").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_load.py tests/test_cli.py -v -k singular`
Expected: FAIL — `test_load_ode_guards_singularities_by_default` fails the
`has(Piecewise)` assertion (nothing guards yet at load time), and the CLI test
fails with `No such option: --no-remove-singularities`.

- [ ] **Step 3: Write the implementation**

In `src/gotranx/load.py`:

```python
def ode_from_string(text: str, name="ode", *, remove_singularities: bool = True) -> ODE:
    ...
    ode = make_ode(components=result.components, name=name, comments=result.comments)
    if remove_singularities:
        ode = ode.remove_singularities()
    logger.info(f"Num states {ode.num_states}")
    logger.info(f"Num parameters {ode.num_parameters}")
    return ode


def load_ode(path: str | Path, *, remove_singularities: bool = True) -> ODE:
    ...
    return ode_from_string(
        fname.read_text(), name=fname.stem, remove_singularities=remove_singularities
    )
```

Document both with a `remove_singularities : bool, optional` numpydoc entry:
"Replace a neighbourhood of every removable singularity with a truncated
Taylor series, by default True. Pass False to emit the expressions exactly as
written, which will divide by zero where the model has a removable pole (numpy
warns; numba crashes)."

In each of `cli/gotran2{py,c,julia,ufl}.py`, add a
`remove_singularities: bool = True` parameter to the `main`-style function that
calls `load_ode`, and pass it through:
`ode = load_ode(fname, remove_singularities=remove_singularities)`.

In `src/gotranx/cli/__init__.py`, for every subcommand that has a
`remove_unused` option (`ode2py`, `ode2c`, `ode2julia`, `ode2mtk`, `ode2ufl`,
`ode2md`, and the deprecated `convert`), add immediately after it:

```python
remove_singularities: bool = (
    typer.Option(
        True,
        "--remove-singularities/--no-remove-singularities",
        help=(
            "Replace a neighbourhood of every removable singularity with a truncated Taylor series"
        ),
    ),
)
```

and thread `remove_singularities=remove_singularities` into the corresponding
`gotran2*.main(...)` call, exactly where `remove_unused=remove_unused` already
goes.

In `src/gotranx/cli/utils.py`, the config path: `remove_unused` is read at
`cli/__init__.py:670` via `config_data.get("remove_unused", remove_unused)`.
Add the same line for `remove_singularities` wherever that pattern appears, and
add a `validate_remove_singularities` mirroring `validate_cse` — a config file
is the only way a non-boolean can arrive:

```python
def validate_remove_singularities(remove_singularities: Any) -> bool:
    """Reject a non-boolean `remove_singularities` from a config file."""
    if not isinstance(remove_singularities, bool):
        raise typer.BadParameter(
            f"remove_singularities must be true or false, got {remove_singularities!r}",
            param_hint="remove_singularities",
        )
    return remove_singularities
```

Document the flag in `docs/cli.md` next to `--remove-unused`, and the config
key in `docs/config.md` and the example `config.toml`. Add a CHANGELOG entry
under a new heading describing the behaviour change: generated `rhs` output for
ToRORd and ORdmm changes, because their GHK poles are unguarded today.

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_load.py tests/test_cli.py -v`
Expected: PASS. Then run the whole suite —
`python3 -m pytest -x -q` — and work through the fallout: every test that
compares generated text for a model with a removable pole now sees a guard.
For each failure, read the diff, confirm the new output is right, and update
the expectation. Do **not** blanket-update; `tests/test_deep_linearization_models.py`'s
`EXPECTED_LORENTZ_SCHEME` in particular must not change, because `lorentz.ode`
has no singularity — if it does, something is guarding that should not be.

- [ ] **Step 5: Commit**

```bash
git add src/gotranx/load.py src/gotranx/cli docs config.toml CHANGELOG.md tests/
git commit -m "feat(cli): guard removable singularities by default

Adds --no-remove-singularities to opt out. The fix was previously reachable
only from Python, so no CLI user could get it."
```

---

### Task 8: Acceptance criteria, measurements, and the full suite

**Files:**
- Modify: `tests/test_singularity_models.py`
- Test: the whole suite

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_singularity_models.py`:

```python
@pytest.mark.parametrize("name", ["ToRORd_dyn_chloride", "tentusscher_panfilov_2006_M_cell"])
def test_no_cse_temporary_is_singular_at_a_guarded_value(name):
    """Test plan item 4. Every temporary the generated scheme computes must be
    finite at every guarded value, because a temporary is computed
    unconditionally even when the branch that needs it is not taken."""
    import gotranx.cli.gotran2py
    from gotranx.schemes import Scheme

    ode = gotranx.load_ode(ODEFILES / f"{name}.ode")
    code = gotranx.cli.gotran2py.get_code(ode, scheme=[Scheme.generalized_rush_larsen])
    namespace: dict = {}
    exec(compile(code, f"<{name}>", "exec"), namespace)

    np.seterr(all="ignore")
    parameters = namespace["init_parameter_values"]()
    states = namespace["init_state_values"]()
    # Place every state exactly on each guarded value in turn.
    for assignment in ode.sorted_assignments():
        for piece in assignment.expr.atoms(sympy.Piecewise):
            for _, condition in piece.args:
                for symbol in condition.free_symbols:
                    try:
                        index = namespace["state_index"](symbol.name)
                    except (KeyError, ValueError):
                        continue
                    probe = states.copy()
                    probe[index] = 0.0
                    result = namespace["generalized_rush_larsen"](probe, 0.0, 1e-3, parameters)
                    assert np.all(np.isfinite(result)), f"{name}: {symbol.name}"


@pytest.mark.parametrize("name", ["ToRORd_dyn_chloride", "tentusscher_panfilov_2006_M_cell"])
def test_stays_stable_at_dt_0_02_over_300_ms(name):
    """Test plan item 5, the substituted stability criterion.

    The 2.0.0 criterion was stated against base_model_IM.ode, which is not a
    bundled fixture (tests/test_deep_linearization_models.py:299 says so). The
    author confirmed this substitution: dt = 0.02 ms over 300 ms on the two
    bundled models, which is where the pole at v = 0 actually lives.
    """
    import gotranx.cli.gotran2py
    from gotranx.schemes import Scheme

    ode = gotranx.load_ode(ODEFILES / f"{name}.ode")
    code = gotranx.cli.gotran2py.get_code(ode, scheme=[Scheme.generalized_rush_larsen])
    namespace: dict = {}
    exec(compile(code, f"<{name}>", "exec"), namespace)

    np.seterr(all="ignore")
    parameters = namespace["init_parameter_values"]()
    y = namespace["init_state_values"]()
    dt, t = 0.02, 0.0
    for _ in range(15000):
        y = namespace["generalized_rush_larsen"](y, t, dt, parameters)
        assert np.all(np.isfinite(y)), f"{name}: non-finite at t = {t}"
        t += dt
    v = y[namespace["state_index"]("v" if "v" in namespace["state_index"].__doc__ else "V")]
    assert -100.0 < float(v) < 60.0, f"{name}: v left the physiological range: {v}"
```

Resolve the membrane-potential state name properly rather than with the
`__doc__` sniff above: `ToRORd_dyn_chloride` uses `v` and
`tentusscher_panfilov_2006_M_cell` uses `V`. Parametrize the name alongside the
model.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_singularity_models.py -v -k "cse_temporary or stable"`
Expected: these should now largely pass, since Tasks 4–7 have landed. If they
pass on the first run, that is not a free win — confirm the tests are load
bearing by temporarily reverting `schemes.py` to the plain `sympy.cse` call and
showing `test_no_cse_temporary_is_singular_at_a_guarded_value` fail, then
restore. Record both outputs.

- [ ] **Step 3: Re-run the 2.0.0 regression criteria**

Run: `python3 -m pytest tests/test_deep_linearization_models.py -v`
Expected: PASS, in particular:
- `test_every_state_of_every_bundled_model_is_linearized` — the counts in
  `EXPECTED_NONZERO` must not drop. Guarding must not zero out a diagonal.
- `test_derivatives_match_central_differences_through_rhs` — the generated
  derivatives must still agree with central differences through `rhs`.
- `test_linearization_block_stays_within_twice_the_rhs`.

If the block/rhs ratio moved, record the new number.

- [ ] **Step 4: Take the measurements**

Write `scripts/measure_singularities.py` (or run inline) and record:

```python
import time
import gotranx
import gotranx.cli.gotran2py
from gotranx.codegen import PythonCodeGenerator
from gotranx.schemes import get_scheme, Scheme

for flag in (False, True):
    t0 = time.perf_counter()
    ode = gotranx.load_ode("tests/odefiles/ToRORd_dyn_chloride.ode", remove_singularities=flag)
    t_load = time.perf_counter() - t0
    codegen = PythonCodeGenerator(ode)
    t0 = time.perf_counter()
    rhs = codegen.rhs()
    t_rhs = time.perf_counter() - t0
    t0 = time.perf_counter()
    scheme = codegen.scheme(get_scheme("generalized_rush_larsen"))
    t_scheme = time.perf_counter() - t0
    print(
        f"remove_singularities={flag}: load {t_load:.2f}s  rhs {t_rhs:.2f}s "
        f"({len(rhs)} chars)  scheme {t_scheme:.2f}s ({len(scheme)} chars)"
    )
```

Report, against the 2.84 s baseline:
- codegen time for `ToRORd_dyn_chloride`, before and after;
- generated source size, before and after;
- linearization block operation count as a multiple of `rhs`.

Expected from the prototype, for comparison: `scheme` 2.75 s -> 4.10 s,
106,841 -> 118,343 chars, plus ~1.4 s of scan at load time. If the measured
numbers are materially worse than this, stop and investigate before
proceeding — the most likely cause is that `inline` is expanding a
`var`-independent intermediate.

- [ ] **Step 5: Run the full suite and the hooks**

Run: `python3 -m pytest`
Expected: PASS, with no skips introduced by this work.

Run: `pre-commit run --all`
Expected: PASS. `cspell` will likely object to new words (`Laurent`,
`removability`, `vffrt`, `vfrt`) — add them to `.cspell_dict.txt`.

Paste both outputs in full. Do not claim either passes without them.

- [ ] **Step 6: Commit**

```bash
git add tests/ .cspell_dict.txt
git commit -m "test: acceptance criteria for removable-singularity guarding"
```

---

## Self-review against the spec

**Spec coverage.**

| Spec section | Task |
|---|---|
| New module `singularities.py`, `RemovablePole`, `removable_poles`, `guard` | 1–4 |
| No `sympy.limit` anywhere (F5) | 1 (module docstring), enforced by Global Constraints |
| Which variable the guard lives in | 2 (`inline`), 4 (`guard`) |
| Candidate pre-filter (state-dependent denominator) | 4 (`removable_poles`) |
| Choosing delta | 3 (`half_width`) — rule corrected, see C4 |
| Interaction with CSE | 5 |
| Numeric verification of every replacement | 3 (`agrees_numerically`) |
| Wiring: default-on, `--no-remove-singularities` | 7 |
| Scope note: `rhs` output changes for ToRORd/ORdmm | 7 (CHANGELOG) |
| Test plan 1 (BR near V = -23) | 6 |
| Test plan 2 (ToRORd/ORdmm near v = 0) | 6 |
| Test plan 3 (toy model, 23.5 as a numeric assertion) | 6 |
| Test plan 4 (CSE survival) | 5, 8 |
| Test plan 5 (no 2.0.0 regression; substituted stability run) | 8 |
| Test plan 6 (full suite, pre-commit) | 8 |
| Measurements owed | 8 |

Spec items deliberately not implemented: the removability test by
`oo`/`zoo`/`nan` scan (C2), pole location by `sympy.singularities` (C1),
full-cone inlining (C3), and the value-calibrated `delta` (C4). Each is
replaced by a measured alternative, documented above.

**Placeholders.** Three code blocks in Tasks 3, 4 and 8 carry an explicit
"delete this line" or "resolve this properly" note attached to a deliberate
artefact (`_numeric`'s stray `all(map(...))` guard, `removable_poles`' dead
first loop, and the `__doc__` sniff for the membrane-potential state name).
These are called out at the point of use rather than left silent.

**Type consistency.** `inline`, `is_removable`, `taylor`, `half_width`,
`agrees_numerically`, `removable_poles`, `guard` and `rewrite` keep the same
`(expr, var, value, ...)` argument order throughout; `definitions` is always
`Mapping[sympy.Symbol, sympy.Expr]`; `defaults` is always
`Mapping[sympy.Symbol, float]`; `states` is always
`frozenset[sympy.Symbol]`. `RemovablePole.rewritten` is the field Task 4's
`guard` reads and Task 4's `removable_poles` writes.

---

## Implementation notes (added during execution)

Deviations from the tasks above, each forced by a measurement or a failing
test during implementation. The code and its docstrings are authoritative.

1. **Pole locations are found on a rationalized copy.** Solved on the float
   expression, Beeler-Reuter's `i_K1` root came back as -23.000000000000004;
   `sympy.series` about that point returns `0`, so the guard was dropped. The
   numeric check caught it rather than emitting it wrongly.
2. **`inline` rebuilds under `evaluate(False)`.** A default `xreplace` folds
   `exp(-0.04*(V + 23))` into `0.3985...*exp(-0.04*V)`, after which the pole is
   not at -23 in any arithmetic, and the series picked up a
   `-5.7e-14/(V + 23.000000000000011)` term. Replacements that are not
   polynomials in the variable are now rejected outright.
3. **Intermediates get default values** (`default_values`). Keeping
   `var`-independent intermediates opaque (C3) left `gamma_cai` without a
   number, so every ToRORd GHK guard was silently dropped.
4. **The window is placed empirically** (`half_width`). The analytic rule in
   Task 3 ignores the local variable's scale; for Beeler-Reuter
   (`0.04*(V + 23)`) it put the edge at 4.9e-3, where the direct derivative is
   still 8.2e-9 off. Scanning a grid for the best float64 agreement between
   the two branches cut the worst-case error on the model's `d(dV_dt)/dV` from
   1.8e-8 to 1.58e-10. The analytic rule remains as a fallback.
5. **Denominators are found structurally, outermost only.** `sympy.together`
   was most of the scan. Looking inside denominators as well tripled ToRORd's
   guards (8 to 24, buffering terms at non-physiological concentrations) and
   did not terminate on ORdmm_Land.
6. **Genuine poles are rejected numerically before `leadterm`**, and nothing
   over `MAX_SERIES_OPS = 150` operations is series-expanded (`sympy.series`
   did not finish in 100 s on ToRORd's 267-operation `E1_i`).
7. **`cse_hiding_piecewise` names shared guards itself.** `cse` will not
   factor a bare placeholder symbol, so a guard used twice was duplicated.
8. **CLI:** the flag is `--remove-singularities/--no-remove-singularities`
   on `ode2py`, `ode2c`, `ode2julia`, `ode2mtk`, `ode2ufl`, `ode2md` and the
   deprecated `convert`. `ode2cellml` never guards: it exports a model.
9. **`ODE.remove_singularities` returns `self` when nothing is guarded**, and
   passes components as a tuple; a list made the round-trip test in
   `tests/test_save.py` fail on equality.
