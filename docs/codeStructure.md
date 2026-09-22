# Code structure

PEREGRINE is a Python package, `peregrinepy`, over a tree of C++ kernels,
`src/compute`. Python decides everything about a case and holds the data;
C++ does the arithmetic. There is no binding layer between them: every
kernel is a plain C function compiled on demand for the case that calls it,
and Python calls it through `ctypes` with a struct of array descriptors.

## The Python side, by role

Each package has one job, and its module docstrings say what it is. The
config names a class in most of them by a `name` attribute, and the package
finds it under its base class, so adding a physics, an integrator, a
boundary condition, a model or a plugin is one subclass and no list to
keep in step.

| package | what it is |
|---|---|
| [files](../src/peregrinepy/files) | The config: its sections and defaults, and what a file's values have to be. |
| [mixture](../src/peregrinepy/mixture) | The gas: species from a mechanism, the library or the case, the equation of state, transport and diffusion models, what each needs and fits, and the reactions. It hands the compiler tables to bake in. |
| [simulator](../src/peregrinepy/simulator) | The physics, as a spec the solver interrogates: the arrays and kernels it needs, its boundary conditions, and its graphs. `euler` and `navierStokes`. A simulator owns nothing. |
| [integrators](../src/peregrinepy/integrators) | How a case steps: the Runge-Kutta schemes and dual time, each holding the solver it moves and a controller that sizes its steps. |
| [multiBlock](../src/peregrinepy/multiBlock) | The blocks and their faces, in layers: `topology` (connectivity), `grid` (coordinates), `restart` (the state), `solver` (everything compiled and runnable). The halo exchange between blocks lives here. |
| [backend](../src/peregrinepy/backend) | Where kernels run: the runtime and its C ABI, arrays on a backend, the tables a launch runs over, the toolchain read from the Kokkos install, the cache, and the compiler. |
| [kernel.py](../src/peregrinepy/kernel.py) | One compiled kernel, described by its own C++: what it runs on and over. |
| [graph.py](../src/peregrinepy/graph.py) | What a step does, as device graphs captured once and submitted from then on, with the halo exchange's host steps between them. |
| [plugins](../src/peregrinepy/plugins) | What runs alongside the stepping. |
| [tools](../src/peregrinepy/tools) | The `peregrine` command's subcommands: running a case, and the grid tools, one module each. |
| [mesher](../src/peregrinepy/mesher), [readers](../src/peregrinepy/readers), [writers](../src/peregrinepy/writers) | Meshes made in memory; grids, results and configs read and written. |
| [partition](../src/peregrinepy/partition) | Balancing a grid's blocks over ranks, and conditioning a grid: merging interfaces away, orienting blocks. |
| [interpolation](../src/peregrinepy/interpolation) | Moving a result from one grid to another, for `peregrine interpolate`. |
| [misc](../src/peregrinepy/misc) | The frozen dict, the MPI helpers, the subclass lookup. |

The [config reference](config.md) says how a case is described;
[running](running.md) how a script uses the solver.

## The compute side

`src/compute` is a tree of kernels and the headers they compose from. The
top level is the contract: `abi.hpp` declares the structs Python fills,
`arrays.hpp` the columns a kernel reads a block's arrays through,
`launch.hpp` the launch shapes, `kernel.hpp` the entry point, and
`runtime.cpp` the runtime, which is compiled like a kernel and holds Kokkos.

| directory | what it holds |
|---|---|
| `advFlux` | The advective flux, composed from a `formula` (KEPaEC, fourth-order KEPaEC, Rusanov, HLLC, AUSM+up), a `reconstruct` (piecewise constant, MUSCL), a `limiter` and a shock-capturing `switch`. |
| `diffFlux` | The diffusive flux. |
| `thermo` | The equations of state behind `eos.hpp`, and the state from conserved or primitive variables. |
| `transport` | The transport models, and the `diffusion` and `mixingRule` headers the compiler picks between. |
| `chemistry` | Reaction rates composed from the baked reaction tables, the production rates as a source, and the substepped source. |
| `boundaryConditions` | One header per boundary condition, each with a body per hook it acts at, compiled through `bc.cpp`. |
| `timeIntegration` | Dual time's pseudo system. |
| `utils` | Copies, combinations, the CFL and finite checks, the gradients, the halo pack and unpack. |

A kernel is a function over one item of a table: a cell, a cell face, a
plane behind a block face. It declares its stencil and the range it runs
over in its own source, and names the arrays it reads and writes in its
struct. Python builds the struct's twin from that declaration, so a kernel
that reads an array the case does not have fails at construction, not at
launch.

## How a kernel is compiled

Every kernel is compiled for the case that runs it. The species count, the
halo depth, the precision and on a device the launch bound are constants in
the compiled code, not runtime values, which is a considerable gain for the
equation of state, transport and chemistry kernels. The species data and
the reactions are written into headers and forced into any source that
reaches `species.hpp` or `reactions.hpp`; the equation of state, diffusion
model and mixing rule are forced into any source that reaches their
header, so a kernel body never names the case's models.

A library is keyed by its source, every header it reaches, the toolchain
and the defines, into the [cache](install.md#the-kernel-cache). Ranks on a
node race to the same file and the first to the lock builds it. The
toolchain, compiler and flags, is read from the Kokkos install; PEREGRINE
does not choose a compiler.

## Tests

`tests/` is a pytest suite by area, and every test that builds a solver
starts the runtime through the `my_setup` fixture. The runtime needs
`Kokkos_ROOT`, so:

```
Kokkos_ROOT=/path/to/kokkos pytest tests/simulator
```

`tests/gate` is the bit gate: every case's interior state after its steps
hashes to what this platform's reference says. The references are recorded
at the start of a round with `PG_GATE_RECORD=1` and kept out of the
repository, so a hash that moves is evidence, not a verdict. On macOS run
the suite one directory per process: a single process aborts silently past
a few hundred loaded kernel libraries.

Python is formatted with `black`, C++ with `clang-format` at the LLVM
default; `setup.cfg` carries the flake8 settings.
