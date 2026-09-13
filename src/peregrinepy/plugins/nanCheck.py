import numpy as np
from mpi4py import MPI

from ..mpiComm.mpiUtils import getCommRankSize
from .base import BasePlugin


class NanCheck(BasePlugin):
    """Stop on a non-finite conserved value: every rank logs where its are,
    the state is written if there is a writer, and the run is over."""

    name = "nanCheck"

    def __call__(self, solver):
        comm, rank, size = getCommRankSize()
        bad = np.array([not solver.allFinite()], np.int32)
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
