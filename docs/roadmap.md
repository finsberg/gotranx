# Roadmap

`gotranx` is under active development and there are several features that we want to add in the future.

The plan is to have all the features from the old [gotran](https://github.com/ComputationalPhysiology/gotran) implemented in `gotranx` (and some more). This includes

- [ ] More numerical schemes
    - [x] Forward Euler
    - [x] Generalized Rush Larsen
    - [x] Hybrid Generalized Rush Larsen
    - [ ] Simplified Implicit Euler
- [ ] Code generation for more languages
    - [x] Python
    - [x] C
    - [x] jax (working but missing tests)
    - [ ] C++
    - [ ] Julia (in progress)
    - [ ] CUDA
    - [ ] OpenCL
    - [ ] Rust
    - [ ] Latex
    - [ ] Markdown
- [x] Converters between commonly used ODE markup languages
    - [x] [`Myokit`](https://github.com/myokit/myokit) (still some limited support for unit conversion, see [issue #26](https://github.com/finsberg/gotranx/issues/26))
    - [x] CellML (supported via MyoKit)
- [ ] Add support for lookup tables. A master student has currently implemented support for this in legacy gotran, see https://www.mn.uio.no/ifi/studier/masteroppgaver/bmi/automated-code-generation-for-simulating-cardiac-c.html
- [ ] Better handling of singularities, see ongoing work here https://github.com/finsberg/gotranx/pull/68
    - The Rush-Larsen schemes linearize by differentiating through intermediates,
      which multiplies removable singularities in terms like `x/(exp(x) - 1)`.
      `remove_singularities` should be extended over the linearized block.
    - `sympy.cse` additionally **hoists** subexpressions out of `Piecewise`
      branches, so a guarded singular term can become an unconditionally
      evaluated temporary (verified on the pinned sympy 1.14: the derivative of
      `Piecewise((0, V < -40), ((V+40)/(exp(-(V+40)/10)-1), True))` CSEs to a
      temporary `1/(1 - x0)` that is singular at `V = -40` and is now evaluated
      unconditionally) -- harmless today, since numpy's `where` evaluates both
      branches anyway and in C the IEEE-754 `inf` is discarded by the ternary
      unless the user traps FP exceptions, but it is a second, distinct
      interaction with CSE worth recording alongside the one above.

If you have additional feature requests, please [open an issue](https://github.com/finsberg/gotranx/issues)
