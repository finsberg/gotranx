[![pre-commit](https://github.com/finsberg/gotranx/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/finsberg/gotranx/actions/workflows/pre-commit.yml)
[![CI](https://github.com/finsberg/gotranx/actions/workflows/main.yml/badge.svg)](https://github.com/finsberg/gotranx/actions/workflows/main.yml)
[![github pages](https://github.com/finsberg/gotranx/actions/workflows/pages.yml/badge.svg)](https://finsberg.github.io/gotranx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# gotranx

`gotranx` is the next generation General ODE translator and is an attempt to a full rewrite of [gotran](https://github.com/ComputationalPhysiology/gotran).

The general idea is that you write your ODE in a high level markup language and use `gotranx` to generate code for solving the ODE in different programming languages.

At the moment we only support Python and C, but we plan to support a wide range of programming languages in the future.

`gotranx` uses [`sympy`](https://www.sympy.org/en/index.html) to generate the numerical schemes.

- Source code: https://github.com/finsberg/gotranx
- Documentation: https://finsberg.github.io/gotranx/


## Install
Install with pip
```
python3 -m pip install gotranx
```
or for the development version
```
python3 -m pip install git+https://github.com/finsberg/gotranx
```

## Quick start
Check out the demos in the [documentation](https://finsberg.github.io/gotranx/demos/cli) and the [examples in the repository](https://github.com/finsberg/gotranx/tree/main/examples).


## Road map
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


If you have additional feature requests, please [open an issue](https://github.com/finsberg/gotranx/issues)

## Contributing
Contributions are very welcomed, but please read the [contributing guide](https://finsberg.github.io/gotranx/CONTRIBUTING/) first
