# History

[`gotran`](https://github.com/ComputationalPhysiology/gotran) was developed back in 2012 by [Johan Hake](https://github.com/johanhake) at the [Simula Research Laboratory](https://www.simula.no) with the main purpose of generating efficient code for [models in cardiac electrophysiology](http://www.scholarpedia.org/article/Models_of_cardiac_cell). In particular it was used to take a model from [CellML](https://www.cellml.org), and translating it into an human readable language, with a suffix `.ode` which could be used to generate code originally in Python, Matlab, C and C++.

In 2014, a master student also [implemented support for Latex and CUDA](https://www.duo.uio.no/handle/10852/41993) under the supervision of Johan Hake.

In 2017, Johan Hake left Simula Research Laboratory and [Henrik Finsberg](https://github.com/finsberg/) took over as the core maintainer of `gotran`. He also added support for Julia and Markdown, and [Kristian Hustad](https://github.com/KGHustad) implemented support for OpenCL and added a lot of optimizations to the code generation. in 2023-2024 Kristian Hustad also had a master student together with Xing Cai to implement support for lookup tables in gotran.


## Why a full rewrite of `gotran`?

The original `gotran` project currently contains about 20 000 lines of python code. In addition it heavily depends on another project called [`modelparameters`](https://github.com/ComputationalPhysiology/modelparameters) which in turn depends heavily on [`sympy`](https://www.sympy.org/en/index.html). This dependency on `sympy` was so strong that we were unable to upgrade sympy passed version 1.1.1, which is why we in 2022 decided to [vendor this version](https://github.com/ComputationalPhysiology/modelparameters/pull/7) into `modelparameters`. Furthermore, some [flaky behavior](https://github.com/ComputationalPhysiology/gotran/issues/24) ended up being very annoying and impossible to track down. These two factors were the main cause of a full rewrite of `gotran`.
