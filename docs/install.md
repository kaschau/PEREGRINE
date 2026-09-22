# Installing

PEREGRINE has no build of its own. The runtime and every kernel a case
needs are compiled at first use, with the compiler and flags the Kokkos
install records, and cached. So installing is: a Kokkos install for the
machine, a Python with the dependencies, and two environment variables.

## Kokkos

Install [Kokkos](https://github.com/kokkos/kokkos) for the machine. The
Kokkos build decides everything about where PEREGRINE runs: the backend
(Serial, OpenMP, CUDA or HIP), the device architecture, and the compiler.
Point `Kokkos_ROOT` (or `Kokkos_DIR`) at the install:

```
export Kokkos_ROOT=/path/to/kokkos/install
```

A CUDA build needs `nvcc` on the path, a HIP build `hipcc`. The compiler
and flags are read from what the install recorded, so a kernel is compiled
exactly as Kokkos was. Static and shared Kokkos builds both work: a static
build's archives are carried whole in the runtime, a shared build's
libraries are found where they are. One Kokkos install serves one backend;
keep one install per backend you run and switch `Kokkos_ROOT`.

## Python

Python 3.10 or later. From the checkout, an editable install brings the
dependencies (`numpy`, `mpi4py`, `h5py`, `pyyaml`, `lxml`) and puts
the `peregrine` command on the path:

```
pip install -e .
```

The install is editable because the kernels are compiled from the
checkout's `src/compute` at run time, so the checkout stays where it is.
`mpi4py` builds against the MPI on the path, so on a cluster load the MPI
module first. Extras: `pip install -e ".[tools]"` adds what the grid tools
need, scipy, matplotlib and pymetis; `".[test]"` pytest and Cantera;
`".[examples]"` matplotlib and Cantera. Cantera is not needed to read a
mechanism; ParaView is imported
only by a case that asks for the Catalyst plugin. Without network access
pip cannot fetch the build backend, so use the setuptools already there:
`pip install -e . --no-build-isolation`.

## The kernel cache

Compiled libraries go to `~/.cache/peregrinepy`, or `$PEREGRINE_CACHE`. A
library is keyed by everything it was compiled from: the source and every
header it reaches, the toolchain, and the defines the case bakes in. So a
case pays for its kernels once, a second case with the same species count
and models pays nothing, and an edit to a header rebuilds only the kernels
that reach it. Ranks on one node race to the same file and the first to the
lock builds it. The cache is safe to delete; the next run rebuilds what it
needs.

A cold cache on a cluster can mean a hundred or more compilations for a
full test suite, so keep it between jobs.

## MPI and parallel HDF5

Every rank's h5py, mpi4py and HDF5 have to be built against the same MPI.
Grids and results are written collectively into one file, so a serial HDF5
build will not run a multi-rank case, and two MPIs in one process will
crash it. A parallel h5py is built from source against a parallel HDF5:

```
export CC=mpicc
export HDF5_MPI="ON"
export HDF5_DIR="/path/to/parallel/hdf5"   # if it is not found by default
pip install h5py --no-binary=h5py
```

To check which MPI each is using:

```
python -c "import h5py; print(h5py.get_config().mpi)"          # must be True
python -c "from mpi4py import MPI; print(MPI.get_vendor())"
ldd $(python -c "import h5py,os;print(os.path.dirname(h5py.__file__))")/*.so | grep libmpi
```

`libmpi.12` is the MPICH ABI soname and `libmpi.40` is Open MPI's; if h5py
and mpi4py name different ones, rebuild h5py against the MPI mpi4py uses.
On macOS read the libraries with `otool -L` instead of `ldd`.

A halo exchange straight from device memory (`haloExchange.kind: device`)
needs an MPI that is GPU-aware, which is the installation's to know; the
default stages messages through pinned host memory and works with any MPI.

## Profiling

Kokkos names every launch, so [Kokkos Tools](https://github.com/kokkos/kokkos-tools)
sees each kernel by name. Point `KOKKOS_PROFILE_LIBRARY` at a connector and
run as usual:

```
export KOKKOS_PROFILE_LIBRARY=/path/to/kokkos-tools/kp_kernel_timer.so
mpiexec -n 1 peregrine run peregrine.yaml g.h5
```

The NVTX connector (`kp_nvprof_connector.so`) marks the launches for Nsight
Systems; the ROCm connector does the same for rocprof. Every kernel is
compiled for the case that runs it, so a profile is of that case's species
count, halo depth and models, not a generic build.
