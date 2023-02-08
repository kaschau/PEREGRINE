from abc import ABCMeta
from ..RHS import RHS
from ..consistify import consistify
from ..compute.utils import AEQB
from ..compute.timeIntegration import rk3s1, rk3s2, rk3s3


class rk3:
    """
    S. Gottlieb, C.-W Shu. Total variation diminishing Runge-Kutta
    schemes. Mathematics of Computation, 67(221):73-85, 1998.

    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def step(self, dt):

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

        for blk in self:
            rk3s3(blk, dt)

        consistify(self)

        self.nrt += 1
        self.tme += dt
        self.titme = self.tme

    step.name = "rk3"
    step.stepType = "explicit"
