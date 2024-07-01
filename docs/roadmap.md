# Roadmap

`gotranx` is under active development and there are several features that we want to add in the future.

The plan is to have all the features from the old [gotran](https://github.com/ComputationalPhysiology/gotran) implemented in `gotranx` (and some more). This includes

- [ ] More numerical schemes
    - [x] Forward Euler
    - [ ] Rush Larsen
    - [x] Generalized Rush Larsen
    - [ ] Hybrid Generalized Rush Larsen
    - [ ] Simplified Implicit Euler
    - [ ] Newton's method for implicit schemes
- [ ] Code generation for more languages
    - [x] Python
    - [x] C
    - [ ] C++
    - [ ] Julia
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

If you have additional feature requests, please [open an issue](https://github.com/finsberg/gotranx/issues)
