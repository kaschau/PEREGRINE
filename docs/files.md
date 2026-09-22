# Files

What PEREGRINE reads and writes: the grid, results, the config, and the
small input files some boundaries and plugins take. Grids and results are
HDF5 with an XDMF sidecar, so ParaView opens the `.xmf` directly. Both are
written collectively into one file, which needs a parallel h5py on a
multi-rank run; see [installing](install.md#mpi-and-parallel-hdf5).

## The grid

`g.h5` carries the grid and everything about it that does not change with
the case: the coordinates of every block, how the blocks connect, and any
partitions the grid has been balanced into.

```
g.h5
  totalBlocks                                        attribute
  coordinates_000000/{x,y,z}                         one group per block, (nk, nj, ni)
  connectivity/{neighbor,orientation,bcName}         (totalBlocks, 6)
  connectivity/{periodicRotation,periodicTranslation}   how a periodic moves
  partitions/1x1/rank                                the base grid
  partitions/64x4/rank                               which rank owns each block
```

Coordinates are stored `(nk, nj, ni)`, the order the device holds them in.
A block's six faces are numbered 1 to 6: the low and high i sides, then j,
then k. The connectivity uses GridPro's orientation notation. A periodic
face stores the transform that reaches its partner, a rotation and a
translation, because that is the shape of the grid.

What kind of boundary a face is does not live here. The grid gives a face
a name and the case says what the name means, in `bcValues`; see
[boundary conditions](boundaryConditions.md). A face left unnamed is
interior if it has a neighbor, periodic if it carries a transform, and an
adiabatic slip wall otherwise.

`pg.readers.GridReader(fileName, ranks=None)` reads a grid, every block or
this rank's share of a partition; `pg.writers.GridWriter(mb, fileName)`
writes one from a multiBlock. `peregrine gridpro2pg` and `peregrine icem2pg`
write grids from GridPro and ICEM output, and condition them on the way: every
interface that can be merged away is, and every block is re-indexed so its
longest extent is i.

### Partitions

A grid carries as many partitions side by side as it has been balanced for,
named `<ranks>x<ranksPerNode>`, so one grid runs on 64 ranks of 4 per node
or of 8 without being rebalanced. The two place blocks differently, because
what crosses a node costs more than what stays on one. A run picks the
partition for its rank count, and the one for its ranks per node if the
grid has it, else another layout of the same count. A grid with no partition
for the rank count refuses to run and says how to add one:

```
peregrine partition g.h5 -ranks 64 -ranksPerNode 4
```

Blocks are assigned whole, so the largest block sets a ceiling no assignment
can beat. The load balancer reports when that, rather than the grouping, is
what binds; `peregrine analyze` reports the same for a grid as it is. The
partitioner is `auto` unless `-method` says `greedy` or `metis`.

## Results

A result is one pair of files, `<name>.h5` and `<name>.xmf`, named from the
step or the time it was written at, `q.<nrt>` unless the writer plugin says
otherwise. There is no distinction between a restart and a frame of an
animation: a run writes as many results as it is asked for, and any one of
them can be restarted from or animated through.

```
q.00000042.h5
  nrt, tme                                           when this is from
  primVars, variables, extras                        what each block group holds
  grid                                               the grid file, relative to this one
  config                                             the case, as its yaml
  peregrine, commit, host, ranks, command, written   where it came from
  results_000000/<variable>                          one group per block, one dataset per export variable
  results_000000/<array>                             one dataset per extra
```

The export variables are `rho`, `p`, `u`, `v`, `w`, `T` and every species'
mass fraction, the last species included. The extras are what the
integrator keeps beyond the state, dual time's state one step back, so a
restart continues exactly. Each variable is stored over the cells of a
block of the grid, not of the partition that wrote it, so any partition can
read the result back. A result carries no grid of its own; its `.xmf`
points at the grid it was run on.

`pg.readers.RestartReader(fileName)` reads one, and is the `state` a solver
starts from; `pg.writers.RestartWriter(mb, fileName, gridFile)` writes one,
its name a pattern of the step `n` or the time `t`, which the writer plugin
does for a run.

## The config

In executable mode the config is a yaml file with the sections of the
[config reference](config.md), read over the defaults by
`pg.readers.readConfigFile(path)`; one rank reads it and every rank gets
the text. A result carries the config that wrote it, so a restart needs no
file. `pg.writers.writeConfigFile(config, fileName)` writes one out, and
`peregrine config peregrine.yaml` writes one of every default to start from.

## Boundary profiles

A boundary that reads values may take them from a file instead of
constants, with `profile: <directory>` in its `bcValues` entry. The file
`<directory>/<bcName>_<block>_<face>.npy` holds two arrays saved one after
the other, the plane of primitive values then the plane of conserved values,
each shaped like the face's value plane. See
[boundary conditions](boundaryConditions.md#profiles).
