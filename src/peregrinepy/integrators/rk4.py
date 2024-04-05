from abc import ABCMeta
from ..RHS import RHS
from ..consistify import consistify
from ..compute.timeIntegration import rk4s1, rk4s2, rk4s3, rk4s4


class rk4:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def step(self, dt):
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

        for blk in self:
            rk4s4(blk, dt)

        consistify(self)

        self.nrt += 1
        self.tme += dt
        self.titme = self.tme

    step.name = "rk4"
    step.stepType = "explicit"
