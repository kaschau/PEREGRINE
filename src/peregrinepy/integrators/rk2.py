from abc import ABCMeta
from ..RHS import RHS
from ..consistify import consistify
from ..compute.utils import AEQB
from ..compute.timeIntegration import rk2s1, rk2s2


class rk2:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def step(self, dt):

        # Stage 1
        self.titme = self.tme
        RHS(self)

        for blk in self:
            rk2s1(blk, dt)

        consistify(self)

        # Stage 2
        self.titme = self.tme + dt
        RHS(self)

        for blk in self:
            rk2s2(blk, dt)

        consistify(self)

        self.nrt += 1
        self.tme += dt
        self.titme = self.tme

    step.name = "rk2"
    step.stepType = "explicit"
