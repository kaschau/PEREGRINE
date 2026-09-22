from pathlib import Path

import numpy as np

from ..misc import getCommRankSize
from .base import BasePlugin


class Trace(BasePlugin):
    """The primitives at chosen cells, appended to one csv per point."""

    name = "trace"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.directory = Path(cfgsect.get("dir", "Trace"))
        with open(cfgsect["points"], "rb") as f:
            self.points = np.load(f)
            self.tags = np.load(f)

    def start(self, solver):
        comm, rank, size = getCommRankSize()
        directory, points, tags = self.directory, self.points, self.tags
        if rank == 0:
            directory.mkdir(parents=True, exist_ok=True)
        comm.Barrier()
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
                    names = blk.primVars
                    fileName.write_text("Time (s), " + ", ".join(names) + "\n")

    def __call__(self, solver):
        # one snapshot per block the traces live in
        names = solver.primVars
        qs = {
            n: solver.exportData(solver.getBlock(n), names)
            for n in {t[0] for t in self.traces}
        }
        for nblki, fileName, i, j, k in self.traces:
            row = [solver.tme] + [qs[nblki][name][i, j, k] for name in names]
            with open(fileName, "a") as f:
                np.savetxt(f, [row], fmt="%.8e", delimiter=",")
