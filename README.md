# PEREGRINE: Accessible, Performant, Portable Multiphysics CFD

<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" width="800" srcset="docs/images/pgSplashD2.jpg">
      <source media="(prefers-color-scheme: light)" width="800" srcset="docs/images/pgSplashL2.jpg">
      <img alt="peregrine logo" width="800" src="docs/images/pgSplashL2.jpg">
    </picture>
</p>

## About

PEREGRINE is a second order, multiblock, structured-grid, finite volume, 3D
multiphysics CFD solver: the Euler and Navier-Stokes equations of a
multi-species gas, with thermally perfect and real-gas equations of state,
kinetic-theory transport, and finite-rate chemistry from a Cantera
mechanism. Its novelty is its implementation: the case is described and
driven in [Python](https://www.python.org), and every kernel is C++ over
[Kokkos](https://github.com/kokkos/kokkos), compiled for the case that runs
it. Kokkos is a C++ library, not a language extension, that abstracts
multidimensional arrays and kernel execution from serial CPU to GPU, so one
source runs a case on a laptop and, without a line changed, on an AMD or
NVIDIA GPU supercomputer. PEREGRINE is parallel across nodes by MPI.

## Installation

Install [Kokkos](https://github.com/kokkos/kokkos) for the machine, point
`Kokkos_ROOT` at the install, and `pip install -e .` from the checkout. The runtime and every kernel a case needs are compiled
at first use, with the compiler and flags the Kokkos install records, into
`~/.cache/peregrinepy`. See [installing](docs/install.md) for the
dependencies, parallel HDF5 and profiling.

## Documentation

Start at the [documentation index](docs/README.md): how to
[run a case](docs/running.md), the [config file](docs/config.md), the
[mixture](docs/mixture.md), [boundary conditions](docs/boundaryConditions.md),
[plugins](docs/plugins.md), the [files](docs/files.md) PEREGRINE reads and
writes, and the [code structure](docs/codeStructure.md).

## Attribution

Please use the following BibTex to cite PEREGRINE in scientific writing:

```
@misc{PEREGRINE,
   author = {Kyle A. Schau},
   year = {2021},
   note = {https://github.com/kaschau/PEREGRINE},
   title = {PEREGRINE: Accessible, Performant, Portable Multiphysics CFD}
}
```

## License

PEREGRINE is released under the New BSD License (see the LICENSE file for details).
Documentation is made available under a Creative Commons Attribution 4.0
license (see <http://creativecommons.org/licenses/by/4.0/>).
