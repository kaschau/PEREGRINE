from functools import partial

import numpy as np
from mpi4py import MPI

from ..kernel import CellCenterKernel
from ..misc import getCommRankSize
from .base import BasePlugin


class NanCheck(BasePlugin):
    """Stops the run on a non-finite conserved value."""

    name = "nanCheck"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        # its own check, compiled with the solver's kernels
        self.kernel = CellCenterKernel("utils/allFinite.cpp")

    def declKernels(self):
        return {"nanCheck allFinite": self.kernel}

    def start(self, solver):
        # settled on the interior of the block table
        self.allFinite = partial(
            self.kernel,
            solver.blockArrayTable,
            solver.blockArrayTable.tiling(self.kernel, "interior"),
        )

    def __call__(self, solver):
        comm, rank, size = getCommRankSize()
        bad = np.array([not self.allFinite()], np.int32)
        comm.Allreduce(MPI.IN_PLACE, bad, op=MPI.SUM)
        if not bad[0]:
            return
        for blk in solver.blocks:
            Q = blk.Q.get()[blk.interior]
            nans = np.where(np.isnan(Q).any(axis=-1))
            if len(nans[0]) == 0:
                continue
            cells = blk.cells.get()[blk.interior]
            with open(f"nans_{blk.nblki}.log", "w") as f:
                f.write(f"Nan Detection Log: Block {blk.nblki}\n")
                for x, y, z in cells[nans]:
                    f.write(f"x = {x} y = {y} z = {z}\n")
        if "writer" in solver.plugins:
            solver.plugins["writer"](solver)
        comm.Barrier()
        raise RuntimeError(
            f"non-finite state at step {solver.nrt}; see the nans_<block>.log files"
        )
