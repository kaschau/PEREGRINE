from abc import ABCMeta
from ..RHS import RHS
from ..consistify import consistify
from ..compute.utils import AEQB
from ..compute.timeIntegration import rk34s1, rk34s2, rk34s3, rk34s4


class rk34:

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def step(self, dt):

        # First Stage
        self.titme = self.tme
        RHS(self)

        for blk in self:
            rk34s1(blk, dt)

        consistify(self)

        # Second Stage
        self.titme = self.tme + dt / 2.0
        RHS(self)

        for blk in self:
            rk34s2(blk, dt)

        consistify(self)

        # Third Stage
        self.titme = self.tme + dt
        RHS(self)

        for blk in self:
            rk34s3(blk, dt)

        consistify(self)

        # Fourth Stage
        self.titme = self.tme + dt / 2.0
        RHS(self)

        for blk in self:
            rk34s4(blk, dt)

        consistify(self)

        self.nrt += 1
        self.tme += dt
        self.titme = self.tme

    step.name = "rk34"
    step.stepType = "explicit"
