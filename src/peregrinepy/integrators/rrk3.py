from abc import ABCMeta

import numpy as np
from scipy.optimize import root_scalar

from ..compute.timeIntegration import rk3s1, rk3s2, rk3s3
from ..compute.utils import sumEntropy, computeEntropy, CEQxApyB
from ..consistify import consistify
from ..RHS import RHS


def getEntropy(g, mb, s0, dsdt):
    # update state based on gamma
    for blk in mb:
        blk.array["Q"][:] = blk.array["Q0"] + g * blk.array["Q1"]
        # update primatives
        mb.eos(blk, mb.thtrdat, -1, "cons")

    # compute new total entropy
    s = computeEntropy(mb)

    residual = s - s0 - g * dsdt
    # print(f"{s-s0 = }, {g*dsdt = }, {residual = }, {g = }")

    return residual


class rrk3:
    """
    Relaxation Runge Kutta

    S. Gottlieb, C.-W Shu. Total variation diminishing Runge-Kutta
    schemes. Mathematics of Computation, 67(221):73-85, 1998.

    """

    __metaclass__ = ABCMeta
    gSol = 1.0

    def __init__(self):
        pass

    def step(self, dt):
        # before we do anything, we need total entropy at un
        s0 = computeEntropy(self)

        # First Stage
        self.titme = self.tme
        RHS(self)

        for blk in self:
            rk3s1(blk, dt)

        consistify(self)

        # Second Stage
        self.titme = self.tme + dt
        RHS(self)

        for blk in self:
            rk3s2(blk, dt)

        consistify(self)

        # Third Stage
        self.titme = self.tme + dt / 2.0
        RHS(self)

        dsdt = 0.0
        for blk in self:
            rk3s3(blk, dt)
            # use Q1,s1 to store \delta t \sum bi*fi
            CEQxApyB(blk.Q1, 1.0, blk.Q, -1.0, blk.Q0)
            CEQxApyB(blk.s1, 1.0, blk.s, -1.0, blk.s0)
        dsdt = sumEntropy(self)

        # now compute gamma
        g = root_scalar(
            getEntropy,
            args=(self, s0, dsdt),
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
            blk.array["Q"][:] = blk.array["Q0"] + self.gSol * blk.array["Q1"]
            blk.array["s"][:] = blk.array["s0"] + self.gSol * blk.array["s1"]

        consistify(self)

        self.nrt += 1
        self.tme += dt * self.gSol
        self.titme = self.tme

    step.name = "rrk3"
    step.stepType = "explicit"
