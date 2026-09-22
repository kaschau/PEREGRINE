from functools import partial
from time import perf_counter

import numpy as np
from mpi4py import MPI

from ..kernel import CellCenterKernel
from ..misc import getCommRankSize
from .base import BasePlugin


class Report(BasePlugin):
    """Everything a run prints; without it a run prints nothing. The CFL is
    reduced over the ranks here, so a run pays for it only when it asks."""

    name = "report"

    banner = (
        " >>> ******************************** <<<\n"
        "              PEREGRINE CFD\n"
        " >>> ******************************** <<<\n"
        "  Copyright (c) 2021-2024 Kyle A. Schau\n"
        "           All rights reserved.\n"
    )

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        # its own reduction, compiled with the solver's kernels
        self.kernel = CellCenterKernel("utils/CFLmax.cpp")

    def declKernels(self):
        return {"report CFLmax": self.kernel}

    def start(self, solver):
        # settled on the interior of the block table
        self.CFLmax = partial(
            self.kernel,
            solver.blockArrayTable,
            solver.blockArrayTable.tiling(self.kernel, "interior"),
        )
        self.started = perf_counter()
        if getCommRankSize()[1] == 0:
            print(self.banner)
            print(solver)

    def before(self, solver, dt):
        """An integrator gathers what it will report only on a step that is
        reported on."""
        solver.integrator.reportDue = self.dueAfter(solver, dt)

    def __call__(self, solver):
        comm, rank, size = getCommRankSize()
        cfl = np.zeros(3, solver.backend.fpdtype)
        self.CFLmax(cfl=cfl)
        comm.Allreduce(MPI.IN_PLACE, cfl, op=MPI.MAX)
        if rank != 0:
            return
        acoustic, convective, both = cfl
        dt = solver.integrator.dt
        print(
            f" >>> --------- nrt: {solver.nrt:<6} ---------- <<<\n",
            f"    tme: {solver.tme:.6E} s\n"
            f"     dt : {dt:.6E} s\n"
            f"     MAX CFL       : {both * dt:.3f}\n"
            f"         Acoustic  : {acoustic * dt:.3f}\n"
            f"         Convective: {convective * dt:.3f}\n"
            " >>> -------------------------------- <<<\n",
        )
        said = solver.integrator.stepReport()
        if said:
            print(said)

    def finalize(self, solver):
        elapsed = perf_counter() - self.started
        cells = solver.numCells
        if getCommRankSize()[1] != 0:
            return
        hrs, rem = divmod(elapsed, 3600.0)
        mins, secs = divmod(rem, 60.0)
        steps = max(solver.nrt, 1)
        print(
            "PEREGRINE simulation completed.\n"
            f"Simulation time: {int(hrs)}h : {int(mins)}m : {int(secs)}s\n"
            f"Seconds/Iteration/Cell: {elapsed / steps / cells:.3e}\n"
        )
