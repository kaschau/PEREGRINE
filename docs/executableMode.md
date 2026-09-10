# Executable Mode (Running real cases)


The directory structure is as follows:

    .myCase
    ├── runPeregrine.py        # symlink to runPeregrine.py
    ├── Results                # Folder to read/write results (q.<nrt>.h5, .xmf)
    ├── Grid                   # Folder to read/write grid (g.h5, g.xmf)
    ├── peregrine.yaml         # PEREGRINE config file (see /src/peregrinepy/files/configFile.py)
    └── Input                  # Folder to hold all input files


## Grid File

`g.h5` carries the grid and everything about it that does not change with the
case: the coordinates of every block, how the blocks connect to each other,
and any partitions the grid has been balanced into.

    g.h5
      totalBlocks                                        attribute
      coordinates_000000/{x,y,z}                         (nk, nj, ni) per block
      connectivity/{neighbor,orientation,bcName}             (totalBlocks, 6)
      connectivity/{periodicRotation,periodicTranslation}  how a periodic moves
      partitions/16/rank                                  which rank owns each

The connectivity uses GridPro notation for block orientation. A grid carries
as many partitions side by side as it has been balanced for, named by the
number of ranks, so one grid runs on 4 or on 64 without being rebalanced.
`utilities/loadBalancer.py` adds one; a run with no partition for its rank
count falls back to one block per rank.

## Boundary Conditions

A face's boundary condition is split between the grid and the case, by what
each one owns. The grid gives a face a **name** and, for a periodic, the
transform that reaches its partner -- the shape of the grid, which does not
change from run to run. The case says what each name **is** and what it
**reads**, in the `bcValues` section of `peregrine.yaml`:

    bcValues:
      theInlet:
        bcType: constantVelocitySubsonicInlet
        u: 12.5
        v: 0.0
        w: 0.0
        T: 415.0

So one grid runs as a wall on one case and an inlet on the next without being
rewritten. Every name the grid carries needs an entry; a bc that reads no
values needs only its `bcType`.

A face the grid leaves unnamed says what it is by itself: one with a neighbor
is `interior`, or periodic if it carries a transform, and one with neither is
an `adiabaticSlipWall`. Name a boundary and define it to make it anything
else. See
[templates](https://github.com/kaschau/PEREGRINE/tree/main/src/peregrinepy/bcs/bcValueTemplates)
for what each kind takes.
