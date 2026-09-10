import numpy as np
from mpi4py import MPI  # noqa: F401

from ..compute.timeIntegration import (
    DTrk3s1,
    DTrk3s2,
    DTrk3s3,
    dQdt,
    localDtau,
    invertDQ,
    residual,
)
from ..compute.utils import AEQB
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
                localDtau(blk.cpp, self.config["RHS"]["diffusion"])

            ##############################################
            # In pseudo time, we integrate primatives
            # so b.Q0 will actually represent primative
            # variable set
            ##############################################

            # Stage 1
            self.titme = self.tme
            RHS(self)
            for blk in self:
                dQdt(blk.cpp, dt)

            # Invert dqdQ, apply first rk stage
            for blk in self:
                invertDQ(blk.cpp, dt, self.thtrdat.cpp, self.config["RHS"]["diffusion"])
                DTrk3s1(blk.cpp)

            consistify(self, "prims")

            # Stage 2
            self.titme = self.tme + dt
            RHS(self)
            for blk in self:
                dQdt(blk.cpp, dt)

            for blk in self:
                invertDQ(blk.cpp, dt, self.thtrdat.cpp, self.config["RHS"]["diffusion"])
                DTrk3s2(blk.cpp)

            consistify(self, "prims")

            # Stage 3
            self.titme = self.tme + dt / 2.0
            RHS(self)
            for blk in self:
                dQdt(blk.cpp, dt)

            for blk in self:
                invertDQ(blk.cpp, dt, self.thtrdat.cpp, self.config["RHS"]["diffusion"])
                DTrk3s3(blk.cpp)

            consistify(self, "prims")

            # Compute residual
            if self.nrt % self.config["io"]["niterPrint"] == 0:
                resid = np.array(residual([blk.cpp for blk in self]), dtype=np.float64)
                comm.Allreduce(MPI.IN_PLACE, resid[0, :], op=MPI.MIN)
                comm.Allreduce(MPI.IN_PLACE, resid[1, :], op=MPI.SUM)
                resid[1, :] = np.sqrt(resid[1, :])
                if rank == 0:
                    printResidual(resid[1, :], nrtDT, self[0].ne)

        ############################################################################
        # End inner, pseudo time loop
        ############################################################################

        # After iterating in pseudo time, shift solution arrays
        for blk in self:
            AEQB(blk.cpp.Qnm1, blk.cpp.Qn)
            AEQB(blk.cpp.Qn, blk.cpp.Q)

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
                        blk.array["Qnm1"][ng:-ng, ng:-ng, ng:-ng, :] = np.load(f)
                        blk.updateDeviceView(["Qnm1"])
                except FileNotFoundError:
                    AEQB(blk.cpp.Qnm1, blk.cpp.Q)
                AEQB(blk.cpp.Qn, blk.cpp.Q)
        else:
            for blk in self:
                AEQB(blk.cpp.Qn, blk.cpp.Q)
                AEQB(blk.cpp.Qnm1, blk.cpp.Q)
