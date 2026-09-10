# Executable Mode (Running real cases)


The directory structure is as follows:

    .myCase
    ├── runPeregrine.py        # symlink to runPeregrine.py
    ├── Archive                # Folder to write archive results (*.h5, *.xmf)
    ├── Restart                # Folder to read/write restarts (q.h5, q.xmf)
    ├── Grid                   # Folder to read/write grid (g.h5, g.xmf)
    ├── peregrine.yaml         # PEREGRINE config file (see /src/peregrinepy/files/configFile.py)
    ├── Input                  # Folder to hold all input files
    └── └── bcFams.yaml        # Boundary conditions file


## Grid File

`g.h5` carries the grid and everything about it that does not change with the
case: the coordinates of every block, how the blocks connect to each other,
and any partitions the grid has been balanced into.

    g.h5
      totalBlocks                                        attribute
      coordinates_000000/{x,y,z}                         (nk, nj, ni) per block
      connectivity/{neighbor,orientation,bcType,bcFam}    (totalBlocks, 6)
      partitions/16/rank                                  which rank owns each

The connectivity uses GridPro notation for block orientation. A grid carries
as many partitions side by side as it has been balanced for, named by the
number of ranks, so one grid runs on 4 or on 64 without being rebalanced.
`utilities/loadBalancer.py` adds one; a run with no partition for its rank
count falls back to one block per rank.

## Boundary Conditions File

The boundary conditions file `bcFams.yaml` specifies the boundary conditions. See [templates](https://github.com/kaschau/PEREGRINE/tree/main/src/peregrinepy/bcs/bcFamTemplates). Boundary values belong to the case, not to the grid, so they stay out of `g.h5`.
