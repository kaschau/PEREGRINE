import numpy as np
from mpi4py import MPI  # noqa: F401

from ..kernels.timeIntegration import (
    DTrk3s1,
    DTrk3s2,
    DTrk3s3,
    dQdt,
    invertDQ,
    localDtau,
    residual,
)
from ..kernels.utils import AEQB
from ..consistify import consistify
from ..mpiComm.mpiUtils import getCommRankSize
from ..RHS import RHS


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


class dualTime:
    integratorName = "dualTime"
    stepType = "dualTime"
    # the inner pseudo time loop is rk3 like
    nStorage = 2

    def step(self, dt):
        comm, rank, size = getCommRankSize()

        ############################################################################
        # Inner, pseudo time loop
        ############################################################################

        # At this point we assume that Qn and Qnm1 are appropriately populated
        # Inner time loop integrating in pseudo time
        for nrtDT in range(20):
            # Determine dtau
            for blk in self:
                localDtau(blk, self.config["RHS"]["diffusion"])

            ##############################################
            # In pseudo time, we integrate primatives
            # so b.Q0 will actually represent primative
            # variable set
            ##############################################

            # Stage 1
            self.titme = self.tme
            RHS(self)
            for blk in self:
                dQdt(blk, dt)

            # Invert dqdQ, apply first rk stage
            for blk in self:
                invertDQ(blk, self.thtrdat, dt, self.config["RHS"]["diffusion"])
                DTrk3s1(blk)

            consistify(self, "prims")

            # Stage 2
            self.titme = self.tme + dt
            RHS(self)
            for blk in self:
                dQdt(blk, dt)

            for blk in self:
                invertDQ(blk, self.thtrdat, dt, self.config["RHS"]["diffusion"])
                DTrk3s2(blk)

            consistify(self, "prims")

            # Stage 3
            self.titme = self.tme + dt / 2.0
            RHS(self)
            for blk in self:
                dQdt(blk, dt)

            for blk in self:
                invertDQ(blk, self.thtrdat, dt, self.config["RHS"]["diffusion"])
                DTrk3s3(blk)

            consistify(self, "prims")

            # Compute residual
            if self.nrt % self.config["io"]["niterPrint"] == 0:
                perBlock = [residual(blk) for blk in self]
                resid = np.array(
                    [
                        np.max([r[0] for r in perBlock], axis=0),
                        np.sum([r[1] for r in perBlock], axis=0),
                    ]
                )
                comm.Allreduce(MPI.IN_PLACE, resid[0, :], op=MPI.MAX)
                comm.Allreduce(MPI.IN_PLACE, resid[1, :], op=MPI.SUM)
                resid[1, :] = np.sqrt(resid[1, :])
                if rank == 0:
                    printResidual(resid[1, :], nrtDT, self[0].ne)

        ############################################################################
        # End inner, pseudo time loop
        ############################################################################

        # After iterating in pseudo time, shift solution arrays
        for blk in self:
            AEQB(blk.Qnm1, blk.Qn)
            AEQB(blk.Qn, blk.Q)

        self.nrt += 1
        self.tme += dt
        self.titme = self.tme

    def initializeDualTime(self):
        # Set Qn
        if self.nrt != 0:
            path = self.config["io"]["resultsDir"]
            for blk in self:
                ng = blk.ng
                fileName = f"{path}/Qnm1.{self.nrt:08d}.{blk.nblki:06d}.npy"
                try:
                    with open(fileName, "rb") as f:
                        Qnm1 = blk.Qnm1.get()
                        Qnm1[ng:-ng, ng:-ng, ng:-ng, :] = np.load(f)
                        blk.Qnm1.set(Qnm1)
                except FileNotFoundError:
                    AEQB(blk.Qnm1, blk.Q)
                AEQB(blk.Qn, blk.Q)
        else:
            for blk in self:
                AEQB(blk.Qn, blk.Q)
                AEQB(blk.Qnm1, blk.Q)
