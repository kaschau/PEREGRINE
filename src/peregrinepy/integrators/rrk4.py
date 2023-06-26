from abc import ABCMeta
from mpi4py import MPI

import numpy as np
from scipy.optimize import root_scalar

from ..compute.timeIntegration import rk4s1, rk4s2, rk4s3, rk4s4
from ..compute.utils import sumEntropy, computeEntropy, CEQxApyB, AEQB
from ..consistify import consistify
from ..RHS import RHS
from ..mpiComm.mpiUtils import getCommRankSize


def getEntropy(g, mb, s0, dsdt):
    # update state based on gamma
    for blk in mb:
        CEQxApyB(blk.Q, 1.0, blk.Q0, g, blk.Q1)
        # update primatives
        mb.eos(blk, mb.thtrdat, -1, "cons")

    # compute new total entropy
    s = np.array([computeEntropy(mb, mb.thtrdat)])
    comm, rank, size = getCommRankSize()
    comm.Allreduce(MPI.IN_PLACE, s, op=MPI.SUM)

    residual = s[0] - s0 - g * dsdt
    # print(f"{s-s0 = }, {g*dsdt = }, {residual = }, {g = }")

    return residual


class rrk4:
    """ """

    __metaclass__ = ABCMeta
    gSol = 1.0

    def __init__(self):
        pass

    def step(self, dt):
        # before we do anything, we need total entropy at un
        comm, rank, size = getCommRankSize()
        s0 = np.array([computeEntropy(self, self.thtrdat)])
        comm.Allreduce(MPI.IN_PLACE, s0, op=MPI.SUM)

        # store zeroth stage solution
        for blk in self:
            AEQB(blk.Q0, blk.Q)

        # First Stage
        self.titme = self.tme
        RHS(self)

        for blk in self:
            rk4s1(blk, dt)

        consistify(self)

        # Second Stage
        self.titme = self.tme + dt / 2
        RHS(self)

        for blk in self:
            rk4s2(blk, dt)

        consistify(self)

        # Third Stage
        self.titme = self.tme + dt / 2
        RHS(self)

        for blk in self:
            rk4s3(blk, dt)

        consistify(self)

        # Fourth Stage
        self.titme = self.tme + dt
        RHS(self)

        dsdt = 0.0
        for blk in self:
            rk4s4(blk, dt)
            # use Q1,s1 to store \delta t \sum bi*fi
            CEQxApyB(blk.Q1, 1.0, blk.Q, -1.0, blk.Q0)
            CEQxApyB(blk.s1, 1.0, blk.s, -1.0, blk.s0)

        dsdt = np.array([sumEntropy(self)])
        comm.Allreduce(MPI.IN_PLACE, dsdt, op=MPI.SUM)

        # now compute gamma
        g = root_scalar(
            getEntropy,
            args=(self, s0[0], dsdt[0]),
            method="bisect",
            xtol=2.22e-13,
            bracket=[0.9, 1.1],
            maxiter=200,
            # x0=self.gSol,
            # x1=self.gSol * 0.999999,
        )

        self.gSol = g.root
        # print(f"{self.gSol = }")

        # update new state after gamma is found
        for blk in self:
            CEQxApyB(blk.Q, 1.0, blk.Q0, self.gSol, blk.Q1)
            CEQxApyB(blk.s, 1.0, blk.s0, self.gSol, blk.s1)

        consistify(self)

        self.nrt += 1
        self.tme += dt * self.gSol
        self.titme = self.tme

    step.name = "rrk4"
    step.stepType = "explicit"
