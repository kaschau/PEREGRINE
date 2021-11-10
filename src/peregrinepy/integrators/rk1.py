from abc import ABCMeta
from ..RHS import RHS
from ..consistify import consistify
from ..compute.utils import ApEQxB


class rk1:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def step(self, dt):

        RHS(self)

        # add it to current solution
        for blk in self:
            # Q = dt * dQ
            ApEQxB(blk.Q, dt, blk.dQ)

        self.nrt += 1
        self.tme += dt

        consistify(self)

    step.name = "rk1"
