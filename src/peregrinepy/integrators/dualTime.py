from abc import ABCMeta

import numpy as np
from mpi4py import MPI

from ..compute.timeIntegration import (
    DTrk3s1,
    DTrk3s2,
    DTrk3s2,
    dQdt,
    invertDQ,
    residual,
)
from ..compute.utils import AEQB, CFLmax
from ..consistify import consistify
from ..mpiComm.mpiUtils import getCommRankSize
from ..RHS import RHS


def printResidual(resid, nrt, ne):
    if nrt == 0:
        string = " SubIter      p          u          v          w          T"
        if ne > 5:
            string += "          Y_1 ... Y_NS"
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
            cfl = np.array(CFLmax(self), dtype=np.float64)
            comm.Allreduce(MPI.IN_PLACE, cfl, op=MPI.MAX)
            # For now, set dtau for a combined acoustic/convective CFL=0.5
            dtau = 0.5 / cfl[2]

            ##############################################
            # In pseudo time, we integrate primatives
            # so b.Q0 will actually represent primative
            # variable set
            ##############################################

            # store zeroth stage solution
            for blk in self:
                AEQB(blk.Q0, blk.q)

            # Stage 1
            for blk in self:
                dQdt(blk, dt)
            RHS(self)

            # Invert dqdQ, apply first rk stage
            for blk in self:
                invertDQ(blk, dt, dtau, self.thtrdat)
                DTrk3s1(blk, dtau)

            consistify(self, "prims")

            # Stage 2
            for blk in self:
                dQdt(blk, dt)
            RHS(self)

            for blk in self:
                invertDQ(blk, dt, dtau, self.thtrdat)
                DTrk3s2(blk, dtau)

            consistify(self, "prims")

            # Stage 3
            for blk in self:
                dQdt(blk, dt)
            RHS(self)

            for blk in self:
                invertDQ(blk, dt, dtau, self.thtrdat)
                DTrk3s2(blk, dtau)

            consistify(self, "prims")

            # Compute residual
            resid = np.array(residual(self), dtype=np.float64)
            comm.Allreduce(MPI.IN_PLACE, resid, op=MPI.MAX)
            if rank == 0:
                printResidual(resid, nrtDT, self[0].ne)

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
        for blk in self:
            AEQB(blk.Qn, blk.Q)
            AEQB(blk.Qnm1, blk.Q)
