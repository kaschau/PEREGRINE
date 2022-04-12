from abc import ABCMeta
from ..RHS import RHS
from ..consistify import consistify
from ..compute.utils import AEQB
from ..compute.timeIntegration import rk4s1, rk4s2, rk4s3, rk4s4


class rk4:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def step(self, dt):

        # store zeroth stage solution
        for blk in self:
            AEQB(blk.rhs0, blk.Q)

        # First Stage
        RHS(self)

        for blk in self:
            rk4s1(blk, dt)

        consistify(self)

        # Second Stage
        RHS(self)

        for blk in self:
            rk4s2(blk, dt)

        consistify(self)

        # Third Stage
        RHS(self)

        for blk in self:
            rk4s3(blk, dt)

        consistify(self)

        # Fourth Stage
        RHS(self)

        for blk in self:
            rk4s4(blk, dt)

        consistify(self)

        self.nrt += 1
        self.tme += dt

    step.name = "rk4"
    step.stepType = "explicit"
