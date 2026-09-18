"""The gate across ranks: the same case through a grid file partitioned
for this many ranks hashes to the one-rank reference, whatever holds
which block.

    python -m tests.gate.ranks make <dir> <ranks>       the grid and its partition
    mpiexec -n <ranks> python -m tests.gate.ranks run <dir> <case>
"""

import sys
from pathlib import Path

import peregrinepy as pg
from mpi4py import MPI
from peregrinepy.backend import abi
from peregrinepy.partition import getPartitioner

from . import cases
from .test_gate import load


def make(directory, ranks):
    """Writes the gate's box as a grid file with a partition for :ranks:."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mb = pg.multiBlock.grid()
    cases.mesh().fill(mb)
    writer = pg.writers.GridWriter(mb, str(directory), precision="double")
    writer.write(mb)
    topology = pg.multiBlock.topology.fromGrid(f"{directory}/g.h5")
    groups = getPartitioner().partition(topology, ranks, ranks)
    writer.writePartition(topology, groups, ranks)


def run(directory, name):
    comm = MPI.COMM_WORLD
    abi.lib.initialize()
    mb = cases.build(
        **cases.cases[name], ranks=(comm.size, comm.size), grid=f"{directory}/g.h5"
    )
    cases.step(mb)
    mine = cases.blockDigests(mb)
    every = comm.gather(mine, root=0)
    if comm.rank == 0:
        got = cases.digest({k: v for d in every for k, v in d.items()})
        platform = cases.platform()
        want = load().get(platform, {}).get(name)
        verdict = "matches" if got == want else f"differs from {want}"
        print(f"{name} on {comm.size} ranks ({platform}): {got} {verdict}", flush=True)
    abi.lib.finalize()


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2], int(sys.argv[3]))
    else:
        run(sys.argv[2], sys.argv[3])
