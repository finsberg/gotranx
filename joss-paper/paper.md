---
title: 'gotranx: General ODE translator'
tags:
  - Python
  - code generation
  - ordinary differential equations
  - parser
authors:
  - name: Henrik Finsberg
    orcid: 0000-0001-6489-8858
    corresponding: true
    affiliation: 1
  - name: Johan Hake
    corresponding: false
    affiliation: 2
affiliations:
 - name: Simula Research Laboratory, Oslo, Norway
   index: 1
 - name: Oslo Katedralskole, Oslo, Norway
   index: 2
date: 20 March 2024
bibliography: paper.bib

---

# Summary

We introduce `gotranx`, a General ODE Translator for automatic code generation of ordinary differential equations (ODEs). The user writes the ODE in a markup language with the file extension `.ode` and the tool generates code with numerical schemes for solving the ODE in different programming languages.

`gotranx` implements a domain specific language (DSL) using [Lark](https://github.com/lark-parser/lark) for representing ODEs. It can parse the variables and equations into a symbolic representation [@meurer2017sympy], and generate numerical schemes based on codeprinters from Sympy, in particular the Rush-Larsen scheme [@rush1978practical] which is very popular in the field of computational biology.

`gotranx` is a full re-implementation of [`gotran`](https://github.com/ComputationalPhysiology/gotran), and the long term goal is to implement the same features in `gotranx` as found in `gotran` along with additional features.

# Statement of need

Systems of ordinary Differential Equations (ODEs) are equations of the form
$$
y'(t) = f(t, y),
$$
where $t$ represents time and $y$ is a vector of state variables. These equations are a fundamental concept in mathematics and science, and plays a critical role in various fields such as physics, engineering, and economics.

Solving ODEs can be done analytically in simple cases but for most real world applications one needs to apply numerical methods [@ascher1998computer]. For this there exist a suite of well established softwares such as `scipy`[@virtanen2020scipy] for Python, `Differentialequations.jl`[@rackauckas2017differentialequations] for Julia and Sundials[@hindmarsh2005sundials] for `C` and `C++` .

Computational biology is one field where ODEs play an essential role, for example in models of heart cells. The resulting ODEs can, in these cases, be quite involved with many parameters and state variables. For example one of the more recent models of human heart cells [@tomek2019development] have 112 parameters, 45 state variables and 276 intermediate variables. Solutions of the state equations of these membrane models typically allows for specialized exponential integrators [@rush1978practical] that are implemented as a numerical scheme in `gotranx`.

Furthermore, such models of a single heart cell are typically embedded into a spatial model, such as the Monodomain or Bidomain model [@sundnes2007computing], where each point in a 3D geometry represents a cell. This means that thousands or even millions of such ODEs needs to be coupled and solved in a larger systems. In these cases, traditional solvers are usually unsuitable and custom code are often developed to solve the ODEs. An alternative to this approach is to use an existing framework specialized for these types of problems [@cooper2020chaste; @plank2021opencarp]. However, introducing a big framework might not be ideal if the user wants to do a lot of customizations. With `gotranx` you can easily generate framework independent code that can easily be integrated into most simulation pipelines. One example one example is [`fenics-beat`](https://github.com/finsberg/fenics-beat) which uses `gotranx` to generate code solving ODEs in a Monodomain model.

While such a models are typically developed in one programming language, different research groups use different programming languages to integrate and solve their models. To make these models programming language independent, it is common practice to write them in a markup language [@lloyd2004cellml; @keating2020sbml]. However, when translating the models to a new programming language, user typically need to do this manually which are likely to introduce errors in the code. Since `gotranx` also implements converters for e.g CellML through MyoKit [@clerx2016myokit], it already supports most models that are used today.


# Acknowledgements
Henrik Finsberg is the main developer of `gotranx`. The original gotran library was created by Johan Hake, and he is acknowledged with co-authorship for this.
We would also like to acknowledge Kristian Hustad for valuable discussions and contributions to the original gotran library.


# References
