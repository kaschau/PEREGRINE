# Running a case

There are two ways to run PEREGRINE. A script builds the case in Python and
steps it, which suits small studies and anything that wants to look at the
state as it goes; [examples/](../examples) is a set of them. Executable mode
runs a case from a config file and a grid file, or from a result, with
the `peregrine` command, which is how production cases run.

## In a script

```python
from mpi4py import MPI  # first, so MPI is up before the runtime
import peregrinepy as pg

# the case
config = pg.files.configFile()
config["simulation"]["simulator"] = "navierStokes"
config["simulation"]["dt"] = 1e-6
config["mixture"]["species"] = ["O2", "N2"]
config["mixture"]["eos"] = "tpg"
config["mixture"]["trans"] = "kineticTheory"
config["mixture"]["Trange"] = (200.0, 1000.0)
config["bcValues"]["still"] = {"bcType": "adiabaticNoSlipWall"}
config["bcValues"]["moving"] = {"bcType": "adiabaticMovingWall", "u": 5.0, "v": 0.0, "w": 0.0}
config.validateConfig()

# the mesh: a box of one block, its j sides named, periodic in i and k
mesh = pg.mesher.CubeMesher(
    mbDims=[1, 1, 1],
    dimsPerBlock=[2, 50, 2],
    lengths=[0.001, 0.025, 0.001],
    periodic=[True, False, True],
    boundaryNames={3: "still", 4: "moving"},
)

pg.backend.abi.lib.initialize()
mb = pg.multiBlock.solver(config, mesh)
print(mb)
for _ in range(1000):
    mb.integrator.step(config["simulation"]["dt"])
data = mb.exportData(mb.blocks[0], ["u", "T"])
pg.backend.abi.lib.finalize()
```

What the pieces are:

- **The config** says everything about the case but its shape. Every key is
  in the [config reference](config.md). `validateConfig()` casts and checks
  what a file would have had checked on reading.
- **The mesh** is anything that fills a multiBlock with blocks, coordinates
  and connectivity: a mesher, or a grid file's reader. `CubeMesher` cuts a
  box into a lattice of `mbDims` blocks of `dimsPerBlock` nodes each;
  `AnnulusMesher` does the same for a wedge of an annulus. `boundaryNames`
  names the outside faces by side, 1 and 2 the low and high i sides, 3 and 4
  j, 5 and 6 k, for `bcValues` to say what each name is; a side left unnamed
  is a slip wall, and a periodic axis has no outside on it.
- **The runtime** is started once per process, before any solver, and
  finalized at the end. It loads the compiled runtime and starts Kokkos on
  this rank's device; the ranks sharing a node take its devices in turn.
  Nothing before `initialize()` touches a device, so a script that only
  reads or writes files never needs one.
- **The solver**, `pg.multiBlock.solver(config, mesh, state=None)`, is the
  case ready to step: the blocks, every kernel compiled, the halo
  exchanges, the boundary conditions, the graphs and the integrator. With
  no `state` it starts uniform at the config's initial conditions; with a
  `RestartReader` it starts from that result.

The solver is the one handle a script holds:

| | |
|---|---|
| `mb.integrator.step(dt)` | One step of `dt`. `mb.nrt` and `mb.tme` move on. |
| `mb.integrator.run()` | `simulation.niter` steps sized by the controller, with every plugin acting as often as it says. |
| `mb.blocks`, `mb.getBlock(n)` | This rank's blocks; a block's arrays are attributes, `blk.Q.get()` a host copy of the conserved state over every cell, halos included. |
| `mb.exportData(blk, names)` | Host arrays of the named variables of a block: `rho`, `p`, `u`, `v`, `w`, `T` and every species. |
| `mb.setPrimitives(primitives)` | Sets the state from one host array per block of the primitive vector, `(p, u, v, w, T, Y...)` over every cell, and makes it consistent. |
| `mb.primVars`, `mb.exportVars` | The primitive vector's names, and everything a result writes. |
| `print(mb)` | The case as the report plugin prints it. |

The script runs under MPI like any other: `mpiexec -n 4 python -m mpi4py
case.py`. A mesher gives every rank the whole lattice, so a multi-rank
script wants a partitioned grid file instead; see below.

## Executable mode

The `peregrine` command, which the install puts on the path, runs a case
from files; `python -m peregrinepy` is the same thing:

```
peregrine run peregrine.yaml g.h5
peregrine run -r q.00000100.h5
mpiexec -n 64 peregrine run peregrine.yaml g.h5
```

The first form takes a config and a grid, and starts from the config's
initial conditions. The second restarts from a result, which carries the
config it was written with and the grid it sits on; a config or a grid given
beside `-r` wins over the result's. The run takes `simulation.niter` steps
and prints nothing unless the config has the report plugin. A failure on
any rank aborts the whole run.

A grid run on more than one rank has to carry a partition for that rank
count, made by `peregrine partition`; the run says so, and how to make one,
if it does not. See [files](files.md#partitions).

Nothing depends on where the files sit: the config names the results
directory, and a result names its grid relative to itself.

## Starting from a state you make

To start a production case from a flow field built in Python, build the
case in a script, set the state, and write a result to restart from:

```python
mb = pg.multiBlock.solver(config, pg.readers.GridReader("g.h5"))
q = [cases_own_field(blk) for blk in mb.blocks]   # (ni+2ng, nj+2ng, nk+2ng, ne) each
mb.setPrimitives(q)
pg.writers.RestartWriter(mb, "q.{n:08d}.h5", "g.h5", precision="double").write(mb)
```

Then `peregrine run -r q.00000000.h5`.

## The tools

The rest of the `peregrine` command works on a case's files. Every input
and output is a file named on the command line, and `peregrine <tool> -h`
says what each takes.

| | |
|---|---|
| `peregrine config peregrine.yaml` | Writes a config of every default to start a case from. |
| `peregrine partition g.h5 -ranks 64 -ranksPerNode 4` | Adds a load-balanced partition to a grid; see [files](files.md#partitions). |
| `peregrine verify g.h5` | Checks a grid's connectivity and that joined faces' coordinates match. |
| `peregrine analyze g.h5` | Reports the block sizes and the partitions a grid carries. |
| `peregrine interpolate q.h5 other.h5 out.h5` | Interpolates a result onto another grid. |
| `peregrine gridpro2pg blk.tmp blk.tmp.conn blk.tmp.pty g.h5` | Translates a GridPro grid. |
| `peregrine icem2pg info.topo g.h5` | Translates an ICEM Multi-Block Info grid. |
| `peregrine channel channel.yaml g.h5` | Makes a channel or boundary layer grid spaced in wall units; `-template` writes the yaml. |
| `peregrine rotate sector.h5 g.h5 -segments 8 -angle 45 -axis 1,0,0` | Copies a sector grid about its axis into more sectors. |

What the tools need beyond the solver, scipy, matplotlib and pymetis, is
the `tools` extra of the install.

## Examples

Every script in [examples/](../examples) is self-contained and says what it
does at the top: one-dimensional advection and diffusion, a two-dimensional
Euler vortex, Taylor-Green on a straight and a skewed box, a normal shock,
shock capturing, Couette flow against its analytical solution, auto-ignition
and a detonation profile against Cantera, real-gas properties, Catalyst
co-processing, and an MPI scaling test.
