# PEREGRINE: Accessible, Performant, Portable Multiphysics CFD

<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" width="800" srcset="https://github.com/kaschau/PEREGRINE/blob/main/docs/images/pgSplashD2.jpg">
      <source media="(prefers-color-scheme: light)" width="800" srcset="https://github.com/kaschau/PEREGRINE/blob/main/docs/images/pgSplashL2.jpg">
      <img alt="peregrine logo" width="800" src="https://github.com/kaschau/PEREGRINE/blob/main/docs/images/pgSplashL2.jpg">
    </picture>
</p>

# About

PEREGRINE is a second order, multiblock, structured-grid multiphysics, finite volume, 3D CFD solver. The main novelty of PEREGRINE is its implementation in [Python](https://www.python.org) for ease of development and use of [Kokkos](https://www.github.com/kokkos/kokkos) for performance portability. If you are unfamiliar with Kokkos, do a little digging, it is a great project with a healthy community and helpful developers. The TLDR; Kokkos is a C++ library (not a C++ language extension) that exposes useful abstractions for data management (i.e. multidimensional arrays) and kernel execution from CPU-Serial to GPU-Parallel. This allows a single source, multiple architecture, approach in PEREGRINE. In other words, you can run a case with PEREGRINE on your laptop, then without changing a single line of source code, run the same case on a AMD GPU based super computer. PEREGRINE is massively parallel inter-node via MPI communication.

# Installation

You must first install [Kokkos](https://www.github.com/kokkos/kokkos) and set the environment variable `Kokkos_DIR=/path/to/kokkos/install`. The Kokkos installation controls the Host/Device + Serial/Parallel execution parameters, there are no settings for the python installation.

## Easy Install
For editable python installation:

``` pip install -e . ```

Note, installation with pip is hard coded to Debug mode. I can't figure out how to make that an option.

## Recommended Install
For development, it is better to set the environment variable `PYTHONPATH` to point to `/path/to/PEREGRINE/src/` followed by manual installation of the C++ `compute` module:

```cd /path/to/PEREGRINE; mkdir build; cd build; ccmake ../; make -j install```

To generate compile_commands.json, 

``` cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../ ```

# Documentation

See the documentation [here](./docs/documentation.md).

# Profiling GPU via NVTX
Download and install the libraries found at [here](https://github.com/kokkos/kokkos-tools). At runtime, ensure the environment variable

    $ export KOKKOS_PROFILE_LIBRARY=/path/to/kokkos-tools/kp_nvprof_connector.so

is set. Finally, run the simulation with nsys enabling cuda,nvtx trace options.

    jsrun -p 1 -g 1 nsys profile -o outPutName --trace cuda,nvtx  -f true --stats=false python -m mpi4py pgScript.py

# Performance

PEREGRINE is pretty fast by default. However, when running a simulation with multiple chemical species, it is recommended to turn on `PEREGRINE_NSCOMPILE` in cmake, and then specify the value of `numSpecies`. This will hard code `ns` at compile time, and gives a considerable performance improvement for EOS/transport calculations.

# Parallel I/O 

Parallel I/O can be achieved with a parallel capable h5py installation. 

    $ export CC=mpicc
    $ export HDF5_MPI="ON"
    $ export HDF5_DIR="/path/to/parallel/hdf5"  # If this isn't found by default
    $ pip install h5py --no-binary=h5py
    
`$HDF5_DIR` must point to a parallel enabled HDF5 installation. Parallel I/O is only applicable when running simulations with `config["io"]["lumpIO"]=true`.

# Attribution

Please use the following BibTex to cite PEREGRINE in scientific writing:

```
@misc{PEREGRINE,
   author = {Kyle A. Schau},
   year = {2021},
   note = {https://github.com/kaschau/PEREGRINE},
   title = {PEREGRINE: Accessible, Performant, Portable Multiphysics CFD}
}
```

# License

PEREGRINE is released under the New BSD License (see the LICENSE file for details).
Documentation is made available under a Creative Commons Attribution 4.0
license (see <http://creativecommons.org/licenses/by/4.0/>).
