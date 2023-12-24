# Data Storage

There are two means of manipulating data: 

1) on the python side as numpy arrays and, 
2) on the C++ side via Kokkos Views. 

Numpy arrays are accessed by a dictionary attribute of the block class called "array", i.e.

    blk.array["q"][i,j,k,l]

Gives access to the primitive variables. On the C++ side, the kokkos views are accesses as members of the same block class, i.e.

    b.q(i,j,k,l)

Note, the Kokkos Views are accessible from the python side, by accessing the block class method as in Kokkos, i.e.

    blk.q

However, you cannot access elements of the Kokkos view on the python side.

## Data Residence

Depending on if you are running a CPU or GPU simulation, the arrays live in different places. Obviously, the Kokkos views exist wherever your execution space is, CPU or GPU. But the python side numpy arrays in the ```blk.array``` dict are always on the host, so they are accessible within python. To facilitate this, we also create a dictionary of mirrors, i.e.

    blk.mirror["q"]

Which is a kokkos mirror view of the main kokkos view. This mirror view always exists on the CPU, and so the ```blk.array``` numpy arrays wrap around these mirror views. So to update the numpy  arrays from kokkos view data on the GPU, simply perform a deep_copy from the kokkos view to the  mirror view. Since the numpy array wraps the mirror view data, the numpy array will be updated with this deep_copy operation. If you are running a CPU case, all these arrays and views point to the same data.

      CPU                     CPU                         CPU/GPU
    array dict --> wraps ( mirror dict ) --> mirrors ( kokkos view )

# Array Names

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
