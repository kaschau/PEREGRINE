# PEREGRINE: Accessible, Performant, Portable Multiphysics CFD

<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" width="800" srcset="docs/images/pgSplashD2.jpg">
      <source media="(prefers-color-scheme: light)" width="800" srcset="docs/images/pgSplashL2.jpg">
      <img alt="peregrine logo" width="800" src="docs/images/pgSplashL2.jpg">
    </picture>
</p>

## About

PEREGRINE is a second order, multiblock, structured-grid multiphysics, finite volume, 3D CFD solver. The main novelty of PEREGRINE is its implementation in [Python](https://www.python.org) for ease of development and use of [Kokkos](https://www.github.com/kokkos/kokkos) for performance portability. If you are unfamiliar with Kokkos, do a little digging, it is a great project with a healthy community and helpful developers. The TLDR; Kokkos is a C++ library (not a C++ language extension) that exposes useful abstractions for data management (i.e. multidimensional arrays) and kernel execution from CPU-Serial to GPU-Parallel. This allows a single source, multiple architecture, approach in PEREGRINE. In other words, you can run a case with PEREGRINE on your laptop, then without changing a single line of source code, run the same case on a AMD GPU based super computer. PEREGRINE is massively parallel inter-node via MPI communication.

## Installation

Install [Kokkos](https://www.github.com/kokkos/kokkos) for the machine -- the Kokkos build decides the backend, the device architecture and the compiler; PEREGRINE has no build of its own -- and point `Kokkos_ROOT` at the install:

```export Kokkos_ROOT=/path/to/kokkos/install```

Put `/path/to/PEREGRINE/src` on `PYTHONPATH`, or `pip install -e .` for an editable install. The runtime and every kernel a case needs are compiled at first use, with the compiler and flags the Kokkos install records, into `~/.cache/peregrinepy` (or `$PEREGRINE_CACHE`). A CUDA build needs `nvcc` on the path.

## Documentation

See the documentation [here](./docs/documentation.md).

## Profiling GPU via NVTX
Download and install the libraries found at [here](https://github.com/kokkos/kokkos-tools). At runtime, ensure the environment variable

    $ export KOKKOS_PROFILE_LIBRARY=/path/to/kokkos-tools/kp_nvprof_connector.so

is set. Finally, run the simulation with nsys enabling cuda,nvtx trace options.

    jsrun -p 1 -g 1 nsys profile -o outPutName --trace cuda,nvtx  -f true --stats=false python -m mpi4py pgScript.py

## Performance

Every kernel is compiled for the case that runs it: the species count and halo depth are constants in the compiled code, not runtime values, which is a considerable gain for the EOS and transport kernels. The compiled kernels are cached (`~/.cache/peregrinepy`, or `$PEREGRINE_CACHE`), so a case pays for its kernels once.

## Parallel I/O 

Parallel I/O can be achieved with a parallel capable h5py installation. 

    $ export CC=mpicc
    $ export HDF5_MPI="ON"
    $ export HDF5_DIR="/path/to/parallel/hdf5"  # If this isn't found by default
    $ pip install h5py --no-binary=h5py
    
`$HDF5_DIR` must point to a parallel enabled HDF5 installation built against the **same MPI as mpi4py**. All output is written to a single collective file, so a serial HDF5 build will not run a multi-rank case, and two MPIs in one process will crash it. To check which each is using:

    $ python -c "import h5py; print(h5py.get_config().mpi)"          # must be True
    $ python -c "from mpi4py import MPI; print(MPI.get_vendor())"
    $ otool -L $(python -c "import h5py,os;print(os.path.dirname(h5py.__file__))")/*.so | grep libmpi

`libmpi.12` is the MPICH ABI soname and `libmpi.40` is Open MPI's; if h5py and mpi4py name different ones, rebuild h5py against the MPI mpi4py uses.

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
