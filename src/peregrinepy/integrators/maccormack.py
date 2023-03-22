from abc import ABCMeta
from ..RHS import RHS
from ..consistify import consistify
from ..compute.utils import AEQB, ApEQxB
from ..compute.timeIntegration import corrector


class maccormack:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def step(self, dt):
        # store zeroth stage solution
        for blk in self:
            AEQB(blk.Q0, blk.Q)

        # Predictor Stage
        self.titme = self.tme
        RHS(self)

        for blk in self:
            ApEQxB(blk.Q, dt, blk.dQ)

        consistify(self)

        # Corrector Stage
        RHS(self)

        for blk in self:
            corrector(blk, dt)

        consistify(self)

        self.nrt += 1
        self.tme += dt
        self.titme = self.tme

    step.name = "maccormack"
    step.stepType = "explicit"
