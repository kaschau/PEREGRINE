from pathlib import Path

import numpy as np

from ..mpiComm.mpiUtils import getCommRankSize
from .base import BasePlugin


class Trace(BasePlugin):
    """The primitives at chosen cells, appended to one csv per point. The
    points are a .npy of (nblki, i, j, k) rows followed by one of tags."""

    name = "trace"

    def __init__(self, solver, cfgsect):
        super().__init__(solver, cfgsect)
        comm, rank, size = getCommRankSize()
        directory = Path(cfgsect.get("dir", "Trace"))
        if rank == 0:
            directory.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

        with open(cfgsect["points"], "rb") as f:
            points = np.load(f)
            tags = np.load(f)

        # (block number, file, i, j, k) of every point on this rank
        self.traces = []
        for blk in solver.blocks:
            ng = blk.ng
            for m in np.flatnonzero(points[:, 0] == blk.nblki):
                i, j, k = points[m, 1:4] + ng
                x, y, z = blk.cells.get()[i, j, k]
                fileName = directory / f"{tags[m]}_{x:.6f}_{y:.6f}_{z:.6f}.csv"
                self.traces.append((blk.nblki, fileName, i, j, k))
                if not fileName.exists():
                    names = ["p", "u", "v", "w", "T"] + blk.speciesNames[:-1]
                    fileName.write_text("Time (s), " + ", ".join(names) + "\n")

    def __call__(self, solver):
        # one snapshot per block the traces live in
        qs = {n: solver.getBlock(n).q.get() for n in {t[0] for t in self.traces}}
        for nblki, fileName, i, j, k in self.traces:
            row = np.concatenate(([solver.tme], qs[nblki][i, j, k, :]))
            with open(fileName, "a") as f:
                np.savetxt(f, [row], fmt="%.8e", delimiter=",")
