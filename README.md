# PEREGRINE

An attempt at a superfast Python/C++ Multi-Physics CFD code using Kokkos for performance portability. 


# Installation


``` setup.py install ```

Or just set PYTHONPATH to point to /path/to/PEREGRINE/src/peregrinepy
followed by manual install

``` mkdir build; cd build; ccmake ../```

To generate compile_commands.json, 

``` cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../ ```


# Arrays


There are two means of manipulating data, on the python side as numpy arrays, and on the C++
side via Kokkos arrays. Python side arrays are accessed by a dictionary attribute
of the block class called "array", i.e.

    blk.array["q"]

Gives access to the primitive variables.

| Name      | Variables          | Index   | Units        |
|:---------:|--------------------|:-------:|--------------|
| **q**     | **Primatives**     |         |              |
|           | pressure           | 0       | Pa           |
|           | u,v,w              | 1,2,3   | m/s          |
|           | temperature        | 4       | K            |
|           | mass fraction      | 5..ne   | []           |
| **Q**     | **Conserved**      |         |              |
|           | density            | 0       | kg/m^3       |
|           | momentum           | 1,2,3   | kg m / s.m^3 |
|           | total energy       | 4       | J/m^3        |
|           | species mass       | 5..ne   | kg/m^3       |
| **qh**    | **Thermo**         |         |              |
|           | gamma              | 0       | []           |
|           | cp                 | 1       | J/kg.K       |
|           | enthalpy           | 2       | J/m^3        |
|           | c                  | 3       | m/s          |
|           | internal energy    | 4       | J/m^3        |
|           | species enthalpy\* | 5..5+ns | J/kg         |
| **qt**    | **Transport**      |         |              |
|           | mu                 | 0       | Pa.s         |
|           | kappa              | 1       | W/m/K        |
|           | D[n]\*             | 2..2+ns | m^2/s        |
| **omega** | **Chemistry**      |         |              |
|           | dTdt               | 0       | K/s          |
|           | d(Yi)dt\*\*        | 1..ns-1 | []/s         |

\*We store all species' enthalpies and diffusion coeff

\*\* While d(Yi)/dt is stored at the end of chemistry, d(rhoYi)/dt is applied to dQ/dt

# Profiling GPU via NVTX
Download and install the libraries found at
``` https://github.com/kokkos/kokkos-tools ```
At runtime, ensure the environment variable
``` export KOKKOS_PROFILE_LIBRARY=$HOME/software/sources/kokkos-tools/kp_nvprof_connector.so```
is set. Finally, run the simulation with nsys enabling cuda,nvtx trace options.
```jsrun -p 1 -g 1 nsys profile -o twelveSpecies30Cubed --trace cuda,nvtx  -f true --stats=false python -m mpi4py threeDTaylorProf.py```

# Performance
PEREGRINE is pretty fast by default. However, when running a simulation, it is recommended to turn on ```PEREGRINE_NSCOMPILE``` in cmake, and then specify the value of ```numSpecies```. This will hard code ```ns``` at compile time, and gives a considerable performance improvement for EOS/transport calculations.

# Parallel I/O 
Parallel I/O can be achieved with a parallel capable h5py installation. 

    $ export CC=mpicc
    $ export HDF5_MPI="ON"
    $ export HDF5_DIR="/path/to/parallel/hdf5"  # If this isn't found by default
    $ pip install h5py --no-binary=h5py
    
``` $HDF5_DIR ``` must point to a parallel enabled HDF5 installation. Parallel I/O is only applicable when running simulations with ```config["io"]["lumpIO"]=true```.

## License

PEREGRINE is released under the New BSD License (see the LICENSE file for details).
Documentation is made available under a Creative Commons Attribution 4.0
license (see <http://creativecommons.org/licenses/by/4.0/>).
