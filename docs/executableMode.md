# Executable Mode (Running real cases)


The directory structure is as follows:

    .myCase
    ├── runPeregrine.py        # symlink to runPeregrine.py
    ├── Archive                # Folder to write archive results (*.h5, *.xmf)
    ├── Restart                # Folder to read/write restarts (q.h5, q.xmf)
    ├── Grid                   # Folder to read/write grid (g.h5, g.xmf)
    ├── peregrine.yaml         # PEREGRINE config file (see /src/peregrinepy/files/configFile.py)
    ├── Input                  # Folder to hold all input files 
    │   ├── conn.yaml          # Connectivity file
    │   ├── bcFams.yaml        # Boundary conditions file
    └── └── blocksForProcs.inp # Load balancing file


## Connectivity File

The connectivity file `conn.yaml` uses GridPro notation for block connectivity.

## Boundary Conditions File

The boundary conditions file `bcFams.yaml` specifies the boundary conditions. See [templates](../src/peregrinepy/bcs/bcFamTemplates).

## Load Balancing File

The load balancing file `blocksForProcs.inp` is just a text file where each line represents the nth MPI rank and which block numbers that rank is responsible for. For example:

```
0,1 # 0th MPI rank has blocks 0,1
2,3 # 1st MPI rank has blocks 2,3
```
