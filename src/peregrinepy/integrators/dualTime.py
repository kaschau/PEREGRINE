import numpy as np
from mpi4py import MPI  # noqa: F401

from ..mpiComm.mpiUtils import getCommRankSize
from .explicit import BaseIntegrator


def printResidual(resid, nrt, ne):
    if nrt == 0:
        string = " SubIter      p          u          v          w          T"
        if ne > 5:
            string += "        Y(1) ... Y(NS-1)"
        print(string)
    string = f"{nrt+1:8d}"
    for n in range(ne):
        string += f" {resid[n]: 1.3E}"
    print(string)


class dualTime(BaseIntegrator):
    integratorName = "dualTime"
    stepType = "dualTime"
    # the inner pseudo time loop is rk3 like
    nStorage = 2
    # the kernels the pseudo time stages call
    sources = (
        "timeIntegration/dualTime.cpp",
        "utils/axpby.cpp",
        "utils/reductions.cpp",
    )

    def setKernels(self):
        """Each kernel the stages call, bound to this solver under its own name."""
        for symbol, kernel in self.jit.load(*self.sources).items():
            setattr(self, symbol[2].lower() + symbol[3:], kernel.bind(self))

    def _stage(self, stage, dt):
        """One pseudo time rk3 stage: the RHS at its time, the physical time
        derivative, the preconditioned update."""
        viscous = self.mb.config["RHS"]["diffusion"]
        self.mb.RHS()
        self.dQdt(dt=dt)
        self.invertDQ(dt=dt, viscous=viscous)
        stage()
        self.mb.consistifyFromPrims()

    def _copy(self, dst, src):
        table = self.mb.table
        self.axpby(A=table.views(dst), a=0.0, b=1.0, B=table.views(src))

    def step(self, dt):
        comm, rank, size = getCommRankSize()
        mb = self.mb

        ############################################################################
        # Inner, pseudo time loop
        ############################################################################

        # At this point we assume that Qn and Qnm1 are appropriately populated
        # Inner time loop integrating in pseudo time
        for nrtDT in range(20):
            # Determine dtau
            self.localDtau(viscous=mb.config["RHS"]["diffusion"])

            ##############################################
            # In pseudo time, we integrate primatives
            # so b.Q0 will actually represent primative
            # variable set
            ##############################################
            mb.titme = mb.tme
            self._stage(self.dTrk3s1, dt)
            mb.titme = mb.tme + dt
            self._stage(self.dTrk3s2, dt)
            mb.titme = mb.tme + dt / 2.0
            self._stage(self.dTrk3s3, dt)

            # Compute residual
            if mb.nrt % mb.config["io"]["niterPrint"] == 0:
                ne = mb[0].ne
                resid = np.zeros((2, ne))
                self.residual(rMax=resid[0], rSum=resid[1])
                comm.Allreduce(MPI.IN_PLACE, resid[0, :], op=MPI.MAX)
                comm.Allreduce(MPI.IN_PLACE, resid[1, :], op=MPI.SUM)
                resid[1, :] = np.sqrt(resid[1, :])
                if rank == 0:
                    printResidual(resid[1, :], nrtDT, ne)

        ############################################################################
        # End inner, pseudo time loop
        ############################################################################

        # After iterating in pseudo time, shift solution arrays
        self._copy("Qnm1", "Qn")
        self._copy("Qn", "Q")
        mb.advance(dt)

    def initialize(self):
        """Qn and Qnm1 from the results directory, or the current state."""
        mb = self.mb
        if mb.nrt != 0:
            path = mb.config["io"]["resultsDir"]
            for blk in mb:
                ng = blk.ng
                fileName = f"{path}/Qnm1.{mb.nrt:08d}.{blk.nblki:06d}.npy"
                try:
                    with open(fileName, "rb") as f:
                        Qnm1 = blk.Qnm1.get()
                        Qnm1[ng:-ng, ng:-ng, ng:-ng, :] = np.load(f)
                        blk.Qnm1.set(Qnm1)
                except FileNotFoundError:
                    blk.Qnm1.copyFrom(blk.Q)
            self._copy("Qn", "Q")
        else:
            self._copy("Qn", "Q")
            self._copy("Qnm1", "Q")
