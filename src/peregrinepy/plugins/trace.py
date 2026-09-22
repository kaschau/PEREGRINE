from pathlib import Path

import numpy as np

from ..misc import getCommRankSize
from .base import BasePlugin
from .cadence import Cadence


def pointOf(spec):
    return np.array([spec["p0"]], dtype=np.float64)


def lineOf(spec):
    p0, p1 = (np.array(spec[k], dtype=np.float64) for k in ("p0", "p1"))
    return p0 + np.linspace(0.0, 1.0, spec["n"])[:, None] * (p1 - p0)


def planeOf(spec):
    p0, p1, p2 = (np.array(spec[k], dtype=np.float64) for k in ("p0", "p1", "p2"))
    a, b = np.meshgrid(
        np.linspace(0.0, 1.0, spec["n01"]),
        np.linspace(0.0, 1.0, spec["n02"]),
        indexing="ij",
    )
    return p0 + a.ravel()[:, None] * (p1 - p0) + b.ravel()[:, None] * (p2 - p0)


makers = {"point": pointOf, "line": lineOf, "plane": planeOf}


class Trace(BasePlugin):
    """The primitive variables at the cells nearest chosen points, appended
    to one csv per trace each time it acts: a point, a line of n points or
    a plane of n01 by n02 points, each with the file it writes."""

    name = "trace"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.specs = cfgsect["traces"]
        for tag, spec in self.specs.items():
            if spec.get("type") not in makers:
                raise ValueError(f"trace {tag}: type is point, line or plane")
            if "file" not in spec:
                raise ValueError(f"trace {tag}: names the file it writes")
        # a trace acts on the section's cadence, or its own when it names one
        self.cadences = {
            tag: Cadence(spec) if {"niterOut", "dtOut"} & set(spec) else self.cadence
            for tag, spec in self.specs.items()
        }

    def due(self, solver):
        return any(cadence.due(solver) for cadence in self.cadences.values())

    def nextTime(self, solver):
        times = [c.next(solver) for c in self.cadences.values()]
        return min((t for t in times if t is not None), default=None)

    def start(self, solver):
        """Finds the cell nearest each point on whichever rank holds it, and
        starts each trace's file with its header unless it is there."""
        comm, rank, size = getCommRankSize()
        # per trace: its file, its points, and this rank's (point, block, i, j, k)
        self.traces = {}
        for tag, spec in self.specs.items():
            points = makers[spec["type"]](spec)
            mine = self._nearest(solver, points, comm)
            self.traces[tag] = (spec["file"], points, mine)
            if rank == 0 and not Path(spec["file"]).exists():
                header = "time, x, y, z, " + ", ".join(solver.primVars) + "\n"
                Path(spec["file"]).write_text(header)

    @staticmethod
    def _nearest(solver, points, comm):
        """This rank's points: the ones whose nearest cell center on any rank
        is on one of its blocks, as (point, block, i, j, k)."""
        best = np.full(len(points), np.inf)
        where = np.zeros((len(points), 4), dtype=int)
        for blk in solver.blocks:
            ng = blk.ng
            cells = blk.cells.get()[blk.interior]
            shape, flat = cells.shape[:3], cells.reshape(-1, 3)
            # |c - p|^2 as |c|^2 - 2 c.p + |p|^2: one matvec per point
            cc = (flat**2).sum(axis=1)
            for n, p in enumerate(points):
                d = cc - 2.0 * (flat @ p) + p @ p
                m = d.argmin()
                if d[m] < best[n]:
                    best[n] = d[m]
                    i, j, k = np.unravel_index(m, shape)
                    where[n] = (blk.nblki, i + ng, j + ng, k + ng)
        # the rank whose cell is nearest keeps the point
        everyone = np.array(comm.allgather(best))
        mine = np.flatnonzero(everyone.argmin(axis=0) == comm.rank)
        return [(int(n), *(int(x) for x in where[n])) for n in mine]

    def __call__(self, solver):
        comm, rank, size = getCommRankSize()
        names = solver.primVars
        for tag, (fileName, points, mine) in self.traces.items():
            if not self.cadences[tag].due(solver):
                continue
            rows, data = [], {}
            for n, nblki, i, j, k in mine:
                if nblki not in data:
                    blk = solver.getBlock(nblki)
                    data[nblki] = (blk.cells.get(), solver.exportData(blk, names))
                cells, values = data[nblki]
                rows.append(
                    (n, [*cells[i, j, k], *(values[v][i, j, k] for v in names)])
                )
            everyone = comm.gather(rows, root=0)
            if rank != 0:
                continue
            rows = sorted((r for perRank in everyone for r in perRank))
            with open(fileName, "a") as f:
                for n, values in rows:
                    f.write(", ".join(f"{v:.8e}" for v in (solver.tme, *values)) + "\n")
