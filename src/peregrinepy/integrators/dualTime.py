from abc import ABCMeta

import numpy as np
from mpi4py import MPI

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
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

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

            # store zeroth stage solution
            for blk in self:
                AEQB(blk.Q0, blk.q)

            # Stage 1
            RHS(self)
            for blk in self:
                dQdt(blk, dt)

            # Invert dqdQ, apply first rk stage
            for blk in self:
                invertDQ(blk, dt, self.thtrdat, self.config["RHS"]["diffusion"])
                DTrk3s1(blk)

            consistify(self, "prims")

            # Stage 2
            RHS(self)
            for blk in self:
                dQdt(blk, dt)

            for blk in self:
                invertDQ(blk, dt, self.thtrdat, self.config["RHS"]["diffusion"])
                DTrk3s2(blk)

            consistify(self, "prims")

            # Stage 3
            RHS(self)
            for blk in self:
                dQdt(blk, dt)

            for blk in self:
                invertDQ(blk, dt, self.thtrdat, self.config["RHS"]["diffusion"])
                DTrk3s3(blk)

            consistify(self, "prims")

            # Compute residual
            if self.nrt % self.config["io"]["niterPrint"] == 0:
                resid = np.array(residual(self), dtype=np.float64)
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
            AEQB(blk.Qnm1, blk.Qn)
            AEQB(blk.Qn, blk.Q)

        self.nrt += 1
        self.tme += dt

    step.name = "dualTime"
    step.stepType = "dualTime"

    def initializeDualTime(self):
        # Set Qn
        if self.nrt != 0:
            animate = self.config["io"]["animateRestart"]
            path = self.config["io"]["restartDir"]
            for blk in self:
                if animate:
                    fileName = f"{path}/Qnm1.{self.nrt:08d}.{blk.nblki:06d}.npy"
                else:
                    fileName = f"{path}/Qnm1.{blk.nblki:06d}.npy"
                try:
                    with open(fileName, "rb") as f:
                        blk.array["Qnm1"][:] = np.load(f)
                        blk.updateDeviceView(["Qnm1"])
                except FileNotFoundError:
                    AEQB(blk.Qnm1, blk.Q)
                AEQB(blk.Qn, blk.Q)
        else:
            for blk in self:
                AEQB(blk.Qn, blk.Q)
                AEQB(blk.Qnm1, blk.Q)
